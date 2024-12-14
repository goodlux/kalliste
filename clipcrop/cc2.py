import os
import shutil
from pathlib import Path
from clipcrop.clipcrop import ClipCrop
from PIL import Image

def get_nearest_larger_aspect_ratio(width, height, target_ratios):
    current_ratio = width / height
    larger_ratios = [r for r in target_ratios if r[0]/r[1] > current_ratio]
    if not larger_ratios:
        return None
    return min(larger_ratios, key=lambda r: abs(r[0]/r[1] - current_ratio))

def expand_bbox_to_ratio(bbox, target_ratio, original_size):
    x1, y1, x2, y2 = bbox
    crop_width, crop_height = x2 - x1, y2 - y1

    target_w, target_h = target_ratio
    target_ratio_float = target_w / target_h
    current_ratio = crop_width / crop_height

    if current_ratio < target_ratio_float:
        # Need to expand width
        new_width = int(crop_height * target_ratio_float)
        extra_width = new_width - crop_width
        x1 = max(0, x1 - extra_width // 2)
        x2 = min(original_size[0], x1 + new_width)
    else:
        # Need to expand height
        new_height = int(crop_width / target_ratio_float)
        extra_height = new_height - crop_height
        y1 = max(0, y1 - extra_height // 2)
        y2 = min(original_size[1], y1 + new_height)

    return (x1, y1, x2, y2)

# Input and output directory paths
input_dir = "/Volumes/p02/2024/2024_MaFiLux_lines_s4_shoot_1_video/"
output_dir = "/Volumes/m01/_cropped_images/"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# Target aspect ratios
target_ratios = [(1, 1), (7, 9), (13, 19), (4, 7), (5, 12), (9, 7), (13, 19), (19, 13), (7, 4), (12, 5)]

# Initialize variables for ClipCrop models
DFE, DM, CLIPM, CLIPP = None, None, None, None

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)

        try:
            # Copy original file to output directory
            base_name, ext = os.path.splitext(filename)
            original_output_path = os.path.join(output_dir, f"{base_name}_original{ext}")
            shutil.copy2(input_path, original_output_path)
            print(f"Saved original file: {original_output_path}")

            # Open the original image
            original_image = Image.open(input_path)
            original_size = original_image.size
            print(f"Original image size: {original_size}")

            # Initialize ClipCrop for the current image
            cc = ClipCrop(input_path)

            # Load models if not already loaded
            if DFE is None:
                DFE, DM, CLIPM, CLIPP = cc.load_models()

            # Extract image using ClipCrop
            result = cc.extract_image(DFE, DM, CLIPM, CLIPP, "woman", num=1)

            if result and isinstance(result[0], dict) and 'image' in result[0]:
                print(f"Processing {filename}")

                # Define clipcrop_image
                clipcrop_image = result[0]['image']

                # Save the ClipCrop bounding box crop with metadata
                clipcrop_output_path = os.path.join(output_dir, f"{base_name}_clipcrop.png")
                clipcrop_image.save(clipcrop_output_path, pnginfo=original_image.info)
                print(f"Saved ClipCrop crop: {clipcrop_output_path}")

                # Get the bounding box from ClipCrop
                clipcrop_bbox = clipcrop_image.getbbox()

                print(f"ClipCrop bounding box (relative to clipcrop image): {clipcrop_bbox}")

                # Get the nearest larger aspect ratio
                crop_width, crop_height = clipcrop_bbox[2] - clipcrop_bbox[0], clipcrop_bbox[3] - clipcrop_bbox[1]
                nearest_larger_ratio = get_nearest_larger_aspect_ratio(crop_width, crop_height, target_ratios)

                if nearest_larger_ratio:
                    # Expand the bounding box to fit the new aspect ratio
                    aspect_bbox = expand_bbox_to_ratio(clipcrop_bbox, nearest_larger_ratio, clipcrop_image.size)
                    aspect_crop = clipcrop_image.crop(aspect_bbox)
                    aspect_output_path = os.path.join(output_dir, f"{base_name}_aspectcrop.png")
                    aspect_crop.save(aspect_output_path, pnginfo=original_image.info)
                    print(f"Saved aspect ratio crop: {aspect_output_path}")
                    print(f"Used aspect ratio: {nearest_larger_ratio[0]}:{nearest_larger_ratio[1]}")
                    print(f"Expanded aspect bbox (relative to clipcrop image): {aspect_bbox}")
                else:
                    print("No larger standard aspect ratio available. Using ClipCrop image as aspect crop.")
                    aspect_output_path = os.path.join(output_dir, f"{base_name}_aspectcrop.png")
                    clipcrop_image.save(aspect_output_path, pnginfo=original_image.info)
                    aspect_crop = clipcrop_image

                print(f"ClipCrop size: {clipcrop_image.size}")
                print(f"Aspect crop size: {aspect_crop.size}")
                print("--------------------")
            else:
                print(f"No subject found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All images processed.")
