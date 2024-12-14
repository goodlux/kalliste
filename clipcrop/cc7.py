import os
from pathlib import Path
from clipcrop.clipcrop import ClipCrop
from PIL import Image
import rawpy
import multiprocessing
from functools import partial
import psutil
import time

# Adjustable parameters
min_score = 0.7  # Minimum confidence score for accepting a detection
prompt = "woman wearing a swimsuit"
input_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Dominican")
output_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Dominican_video_crop")
image_extensions = ('.jpg', '.jpeg', '.png', '.dng')
target_ratios = [(1, 1), (7, 9), (13, 19), (4, 7),  (9, 7), (13, 19), (19, 13), (7, 4) ]

# Resource management parameters
max_cpu_percent = 80  # Maximum CPU usage percentage
max_memory_percent = 80  # Maximum memory usage percentage
num_processes = 2  # Start with a conservative number of processes
batch_size = 10  # Number of images to process in each batch

def get_nearest_larger_aspect_ratio(width, height, target_ratios):
    current_ratio = width / height
    print(f"Current image ratio: {current_ratio:.4f}")
    larger_ratios = []
    for ratio in target_ratios:
        target_ratio = ratio[0] / ratio[1]
        print(f"Checking target ratio {ratio[0]}:{ratio[1]} = {target_ratio:.4f}")
        if target_ratio > current_ratio:
            larger_ratios.append(ratio)

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
    print(f"Original bounding box midpoints: X={original_mid_x:.2f}, Y={original_mid_y:.2f}")

    target_w, target_h = target_ratio
    target_ratio_float = target_w / target_h
    current_ratio = crop_width / crop_height

    if current_ratio < target_ratio_float:
        # Need to expand width
        new_width = int(crop_height * target_ratio_float)
        new_height = crop_height
        # Calculate the midpoint of the original width
        mid_x = (x1 + x2) / 2
        # Expand from the sides of the original bounding box
        x1_new = max(0, int(mid_x - new_width / 2))
        x2_new = min(original_size[0], int(mid_x + new_width / 2))
        y1_new, y2_new = y1, y2  # Height remains unchanged
    else:
        # Need to expand height
        new_width = crop_width
        new_height = int(crop_width / target_ratio_float)
        # Calculate the midpoint of the original height
        mid_y = (y1 + y2) / 2
        # Expand from the top and bottom of the original bounding box
        y1_new = max(0, int(mid_y - new_height / 2))
        y2_new = min(original_size[1], int(mid_y + new_height / 2))
        x1_new, x2_new = x1, x2  # Width remains unchanged

    # Adjust if we've hit the image boundaries
    if x1_new == 0:
        x2_new = min(original_size[0], x1_new + new_width)
    elif x2_new == original_size[0]:
        x1_new = max(0, x2_new - new_width)

    if y1_new == 0:
        y2_new = min(original_size[1], y1_new + new_height)
    elif y2_new == original_size[1]:
        y1_new = max(0, y2_new - new_height)

    # Calculate and print new midpoints
    new_mid_x = (x1_new + x2_new) / 2
    new_mid_y = (y1_new + y2_new) / 2
    print(f"New cropped image midpoints: X={new_mid_x:.2f}, Y={new_mid_y:.2f}")

    return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))

def open_image(file_path):
    if file_path.suffix.lower() == '.dng':
        with rawpy.imread(str(file_path)) as raw:
            return Image.fromarray(raw.postprocess())
    else:
        return Image.open(file_path)

def process_image(file_path, output_dir, min_score, prompt, target_ratios):
    try:
        # Open the original image
        original_image = open_image(file_path)
        original_size = original_image.size
        print(f"Processing {file_path.name}")
        print(f"Original image size: {original_size}")

        # Calculate and print original aspect ratio
        original_aspect_ratio = original_size[0] / original_size[1]
        print(f"Original aspect ratio: {original_aspect_ratio:.4f}")

        # Initialize ClipCrop for the current image
        cc = ClipCrop(str(file_path))

        # Load models (each process will load its own copy)
        DFE, DM, CLIPM, CLIPP = cc.load_models()

        # Extract image using ClipCrop with the specified prompt
        result = cc.extract_image(DFE, DM, CLIPM, CLIPP, prompt, num=1)

        # Process result
        if result and isinstance(result[0], dict) and 'image' in result[0]:
            detection_score = result[0].get('score', 0)
            print(f"Detection score: {detection_score}")
            if detection_score >= min_score:
                clipcrop_image = result[0]['image']
                clipcrop_bbox = clipcrop_image.getbbox()
                print(f"ClipCrop bounding box: {clipcrop_bbox}")

                # Calculate and print ClipCrop aspect ratio
                crop_width, crop_height = clipcrop_bbox[2] - clipcrop_bbox[0], clipcrop_bbox[3] - clipcrop_bbox[1]
                clipcrop_aspect_ratio = crop_width / crop_height
                print(f"ClipCrop aspect ratio: {clipcrop_aspect_ratio:.4f}")

                # Get the nearest larger aspect ratio
                nearest_larger_ratio = get_nearest_larger_aspect_ratio(crop_width, crop_height, target_ratios)

                if nearest_larger_ratio:
                    print(f"Expanding to ratio: {nearest_larger_ratio[0]}:{nearest_larger_ratio[1]}")
                    # Expand the bounding box to fit the new aspect ratio
                    aspect_bbox = expand_bbox_to_ratio(clipcrop_bbox, nearest_larger_ratio, original_image.size)
                    aspect_crop = original_image.crop(aspect_bbox)
                    base_name = file_path.stem
                    aspect_output_path = output_dir / f"{base_name}_aspectcrop.png"
                    aspect_crop.save(aspect_output_path, format='PNG')
                    print(f"Saved aspect ratio crop: {aspect_output_path}")
                    print(f"Used aspect ratio: {nearest_larger_ratio[0]}:{nearest_larger_ratio[1]}")
                    print(f"Final aspect ratio: {nearest_larger_ratio[0] / nearest_larger_ratio[1]:.4f}")
                    print(f"Expanded aspect bbox: {aspect_bbox}")
                else:
                    print("No larger standard aspect ratio available. Using ClipCrop image as aspect crop.")
                    base_name = file_path.stem
                    aspect_output_path = output_dir / f"{base_name}_aspectcrop.png"
                    clipcrop_image.save(aspect_output_path, format='PNG')
                    aspect_crop = clipcrop_image
                    print(f"Final aspect ratio: {clipcrop_aspect_ratio:.4f}")

                print(f"ClipCrop size: {clipcrop_image.size}")
                print(f"Aspect crop size: {aspect_crop.size}")
            else:
                print(f"Detection score below threshold. No crop saved.")
        else:
            print(f"No subject found in {file_path.name}. No crop saved.")
        print("--------------------")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

def process_batch(batch, output_dir, min_score, prompt, target_ratios):
    for file_path in batch:
        process_image(file_path, output_dir, min_score, prompt, target_ratios)

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]

        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > max_cpu_percent or memory_percent > max_memory_percent:
            print(f"System resources high (CPU: {cpu_percent}%, Memory: {memory_percent}%). Waiting...")
            time.sleep(60)  # Wait for a minute before trying again
            continue

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_processes)

        # Create a partial function with fixed arguments
        process_batch_partial = partial(process_batch, output_dir=output_dir, min_score=min_score, prompt=prompt, target_ratios=target_ratios)

        # Process the batch
        pool.apply_async(process_batch_partial, (batch,))

        # Close the pool and wait for the batch to finish
        pool.close()
        pool.join()

        print(f"Batch of {len(batch)} images processed.")

    print("All images processed.")
