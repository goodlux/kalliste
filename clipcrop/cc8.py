import os
from pathlib import Path
from clipcrop.clipcrop import ClipCrop
from PIL import Image
import rawpy

# Adjustable parameters
min_score = 0.7  # Minimum confidence score for accepting a detection
prompt = "woman wearing a swimsuit"
input_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Dominican")
output_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Dominican_video_crop")
image_extensions = ('.jpg', '.jpeg', '.png', '.dng')
target_ratios = [(1, 1), (7, 9), (13, 19), (4, 7), (9, 7), (13, 19), (19, 13), (7, 4)]

def get_nearest_larger_aspect_ratio(width, height, target_ratios):
    current_ratio = width / height
    print(f"Current image ratio: {current_ratio:.4f}")
    larger_ratios = [ratio for ratio in target_ratios if ratio[0]/ratio[1] > current_ratio]
    if not larger_ratios:
        print("No larger standard aspect ratio found.")
        return None
    nearest_ratio = min(larger_ratios, key=lambda r: abs(r[0]/r[1] - current_ratio))
    print(f"Selected nearest larger ratio: {nearest_ratio[0]}:{nearest_ratio[1]} = {nearest_ratio[0]/nearest_ratio[1]:.4f}")
    return nearest_ratio

def expand_bbox_to_ratio(bbox, target_ratio, original_size):
    x1, y1, x2, y2 = bbox
    crop_width, crop_height = x2 - x1, y2 - y1

    # Calculate and print original midpoints
    original_mid_x = (x1 + x2) / 2
    original_mid_y = (y1 + y2) / 2
    print(f"Original bounding box midpoint: X={original_mid_x:.2f}, Y={original_mid_y:.2f}")

    target_w, target_h = target_ratio
    target_ratio_float = target_w / target_h
    current_ratio = crop_width / crop_height

    if current_ratio < target_ratio_float:
        # Need to expand width
        new_width = int(crop_height * target_ratio_float)
        new_height = crop_height
    else:
        # Need to expand height
        new_width = crop_width
        new_height = int(crop_width / target_ratio_float)

    # Expand from the center
    x1_new = max(0, int(original_mid_x - new_width / 2))
    x2_new = min(original_size[0], int(original_mid_x + new_width / 2))
    y1_new = max(0, int(original_mid_y - new_height / 2))
    y2_new = min(original_size[1], int(original_mid_y + new_height / 2))

    # Calculate and print new midpoints
    new_mid_x = (x1_new + x2_new) / 2
    new_mid_y = (y1_new + y2_new) / 2
    print(f"New aspect crop midpoint: X={new_mid_x:.2f}, Y={new_mid_y:.2f}")

    return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))

def open_image(file_path):
    if file_path.suffix.lower() == '.dng':
        with rawpy.imread(str(file_path)) as raw:
            return Image.fromarray(raw.postprocess())
    else:
        return Image.open(file_path)

def process_image(file_path, output_dir, min_score, prompt, target_ratios):
    try:
        original_image = open_image(file_path)
        original_size = original_image.size
        print(f"Processing {file_path.name}")
        print(f"Original image size: {original_size}")

        cc = ClipCrop(str(file_path))
        DFE, DM, CLIPM, CLIPP = cc.load_models()
        result = cc.extract_image(DFE, DM, CLIPM, CLIPP, prompt, num=1)

        if result and isinstance(result[0], dict) and 'image' in result[0]:
            detection_score = result[0].get('score', 0)
            print(f"Detection score: {detection_score}")
            if detection_score >= min_score:
                clipcrop_image = result[0]['image']
                clipcrop_size = clipcrop_image.size

                # Find the position of ClipCrop result in the original image
                for y in range(original_size[1] - clipcrop_size[1] + 1):
                    for x in range(original_size[0] - clipcrop_size[0] + 1):
                        if original_image.crop((x, y, x + clipcrop_size[0], y + clipcrop_size[1])) == clipcrop_image:
                            clipcrop_position = (x, y)
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError("Could not find ClipCrop result in original image")

                print(f"ClipCrop position in original image: {clipcrop_position}")

                # Adjust ClipCrop bounding box to original image coordinates
                clipcrop_bbox = clipcrop_image.getbbox()
                adjusted_bbox = (
                    clipcrop_position[0] + clipcrop_bbox[0],
                    clipcrop_position[1] + clipcrop_bbox[1],
                    clipcrop_position[0] + clipcrop_bbox[2],
                    clipcrop_position[1] + clipcrop_bbox[3]
                )
                print(f"Adjusted bounding box: {adjusted_bbox}")

                crop_width, crop_height = adjusted_bbox[2] - adjusted_bbox[0], adjusted_bbox[3] - adjusted_bbox[1]
                clipcrop_aspect_ratio = crop_width / crop_height
                print(f"ClipCrop aspect ratio: {clipcrop_aspect_ratio:.4f}")

                nearest_larger_ratio = get_nearest_larger_aspect_ratio(crop_width, crop_height, target_ratios)

                if nearest_larger_ratio:
                    print(f"Expanding to ratio: {nearest_larger_ratio[0]}:{nearest_larger_ratio[1]}")
                    aspect_bbox = expand_bbox_to_ratio(adjusted_bbox, nearest_larger_ratio, original_image.size)
                    aspect_crop = original_image.crop(aspect_bbox)
                    base_name = file_path.stem
                    aspect_output_path = output_dir / f"{base_name}_aspectcrop.png"
                    aspect_crop.save(aspect_output_path, format='PNG')
                    print(f"Saved aspect ratio crop: {aspect_output_path}")
                    print(f"Final aspect ratio: {nearest_larger_ratio[0] / nearest_larger_ratio[1]:.4f}")
                    print(f"Expanded aspect bbox: {aspect_bbox}")
                else:
                    print("No larger standard aspect ratio available. Using ClipCrop image as aspect crop.")
                    base_name = file_path.stem
                    aspect_output_path = output_dir / f"{base_name}_aspectcrop.png"
                    clipcrop_image.save(aspect_output_path, format='PNG')
                    aspect_crop = clipcrop_image

                print(f"ClipCrop size: {clipcrop_image.size}")
                print(f"Aspect crop size: {aspect_crop.size}")
            else:
                print(f"Detection score below threshold. No crop saved.")
        else:
            print(f"No subject found in {file_path.name}. No crop saved.")
        print("--------------------")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == '__main__':
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    for file_path in image_files:
        process_image(file_path, output_dir, min_score, prompt, target_ratios)

    print("All images processed.")
