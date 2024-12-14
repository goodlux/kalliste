import os
from pathlib import Path
from clipcrop.clipcrop import ClipCrop
from PIL import Image

def get_expanded_crop_size(image_size, crop_box, zoom_percent=5):
    img_width, img_height = image_size
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]

    # Calculate the expanded crop size
    new_width = min(img_width, int(crop_width * (100 + zoom_percent) / 100))
    new_height = min(img_height, int(crop_height * (100 + zoom_percent) / 100))

    return (new_width, new_height)

def find_best_aspect_ratio(expanded_size, target_ratios=[(1, 1), (7, 9), (13, 19), (4, 7), (5, 12), (9, 7), (13, 19), (19, 13), (7, 4), (12, 5)]):
    width, height = expanded_size
    current_ratio = width / height

    # Find the closest ratio that fits within the expanded crop
    valid_ratios = [r for r in target_ratios if (r[0]/r[1] <= current_ratio and r[1]/r[0] <= current_ratio) or (r[1]/r[0] >= current_ratio and r[0]/r[1] >= current_ratio)]

    if not valid_ratios:
        return min(target_ratios, key=lambda r: abs(r[0]/r[1] - current_ratio))

    return min(valid_ratios, key=lambda r: abs(r[0]/r[1] - current_ratio))

def crop_to_aspect_ratio(image, crop_box, aspect_ratio):
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    target_ratio = aspect_ratio[0] / aspect_ratio[1]

    if crop_width / crop_height > target_ratio:
        new_width = int(crop_height * target_ratio)
        center_x = (crop_box[0] + crop_box[2]) // 2
        left = max(0, center_x - new_width // 2)
        right = min(image.width, left + new_width)
        return image.crop((left, crop_box[1], right, crop_box[3]))
    else:
        new_height = int(crop_width / target_ratio)
        center_y = (crop_box[1] + crop_box[3]) // 2
        top = max(0, center_y - new_height // 2)
        bottom = min(image.height, top + new_height)
        return image.crop((crop_box[0], top, crop_box[2], bottom))

# Input and output directory paths
input_dir = "/Volumes/m01/_vlc_snaps/"
output_dir = "/Volumes/m01/_cropped_images/"

Path(output_dir).mkdir(parents=True, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

DFE, DM, CLIPM, CLIPP = None, None, None, None

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)

        try:
            original_image = Image.open(input_path)
            cc = ClipCrop(input_path)

            if DFE is None:
                DFE, DM, CLIPM, CLIPP = cc.load_models()

            result = cc.extract_image(DFE, DM, CLIPM, CLIPP, "woman", num=1)

            if result and isinstance(result[0], dict) and 'image' in result[0]:
                print(f"Processing {filename}")

                # Get the bounding box of the direct crop
                direct_crop = result[0]['image']
                crop_box = direct_crop.getbbox()

                # Calculate the expanded crop size
                expanded_size = get_expanded_crop_size(original_image.size, crop_box, zoom_percent=5)

                # Find the best aspect ratio based on the expanded size
                best_ratio = find_best_aspect_ratio(expanded_size)

                # Crop the original image to the best aspect ratio
                final_crop = crop_to_aspect_ratio(original_image, crop_box, best_ratio)

                output_path = os.path.join(output_dir, f"cropped_{filename}")
                final_crop.save(output_path)

                print(f"Saved crop: {output_path}")
                print(f"Crop size: {final_crop.size}")
                print(f"Aspect ratio: {best_ratio}")
                print("--------------------")
            else:
                print(f"No 'woman' found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All images processed.")
