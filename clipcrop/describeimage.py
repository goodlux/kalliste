import os
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
        new_width = int(crop_height * target_ratio_float)
        extra_width = new_width - crop_width
        x1 = max(0, x1 - extra_width // 2)
        x2 = min(original_size[0], x1 + new_width)
    else:
        new_height = int(crop_width / target_ratio_float)
        extra_height = new_height - crop_height
        y1 = max(0, y1 - extra_height // 2)
        y2 = min(original_size[1], y1 + new_height)
    
    return (x1, y1, x2, y2)

# Input directory path
input_dir = "/Volumes/m01/vlc_snaps/"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Target aspect ratios
target_ratios = [(1, 1), (7, 9), (13, 19), (4, 7), (5, 12), (9, 7), (13, 19), (19, 13), (7, 4), (12, 5)]

# Initialize variables for ClipCrop models
DFE, DM, CLIPM, CLIPP = None, None, None, None

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Open the original image
            original_image = Image.open(input_path)
            original_size = original_image.size
            print(f"\nProcessing: {filename}")
            print(f"1. Original image dimensions: {original_size[0]}x{original_size[1]}")

            # Initialize ClipCrop for the current image
            cc = ClipCrop(input_path)
            
            # Load models if not already loaded
            if DFE is None:
                DFE, DM, CLIPM, CLIPP = cc.load_models()
            
            # Extract image using ClipCrop
            result = cc.extract_image(DFE, DM, CLIPM, CLIPP, "woman", num=1)
            
            if result and isinstance(result[0], dict) and 'image' in result[0]:
                # Get the ClipCrop bounding box
                clipcrop_bbox = result[0]['bbox']
                
                print(f"2. ClipCrop image: left-top point (x,y): ({clipcrop_bbox[0]}, {clipcrop_bbox[1]}), size: {clipcrop_bbox[2]-clipcrop_bbox[0]}x{clipcrop_bbox[3]-clipcrop_bbox[1]}")
                
                # Get the nearest larger aspect ratio
                crop_width, crop_height = clipcrop_bbox[2] - clipcrop_bbox[0], clipcrop_bbox[3] - clipcrop_bbox[1]
                nearest_larger_ratio = get_nearest_larger_aspect_ratio(crop_width, crop_height, target_ratios)
                
                if nearest_larger_ratio:
                    # Expand the bounding box to fit the new aspect ratio
                    aspect_bbox = expand_bbox_to_ratio(clipcrop_bbox, nearest_larger_ratio, original_size)
                    print(f"3. AspectCrop image: left-top point (x,y): ({aspect_bbox[0]}, {aspect_bbox[1]}), size: {aspect_bbox[2]-aspect_bbox[0]}x{aspect_bbox[3]-aspect_bbox[1]}")
                else:
                    print("3. No larger aspect ratio available. AspectCrop is identical to ClipCrop.")
            else:
                print(f"No 'woman' found in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("\nAnalysis complete.")