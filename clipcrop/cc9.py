import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Adjustable parameters
confidence_threshold = 0.3  # Minimum confidence score for accepting a detection
input_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Lines_s4_shoot1_vid")
output_dir = Path("/Volumes/p02/2024/2024_MaFiLux_Lines_s4_shoot1_vid_crop")
image_extensions = ('.jpg', '.jpeg', '.png')
target_ratios = [(1, 1), (7, 9), (13, 19), (4, 7), (9, 7), (13, 19), (19, 13), (7, 4)]

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' for larger models

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

    original_mid_x = (x1 + x2) / 2
    original_mid_y = (y1 + y2) / 2
    print(f"Original bounding box midpoint: X={original_mid_x:.2f}, Y={original_mid_y:.2f}")

    target_w, target_h = target_ratio
    target_ratio_float = target_w / target_h
    current_ratio = crop_width / crop_height

    if current_ratio < target_ratio_float:
        new_width = int(crop_height * target_ratio_float)
        new_height = crop_height
    else:
        new_width = crop_width
        new_height = int(crop_width / target_ratio_float)

    x1_new = max(0, int(original_mid_x - new_width / 2))
    x2_new = min(original_size[0], int(original_mid_x + new_width / 2))
    y1_new = max(0, int(original_mid_y - new_height / 2))
    y2_new = min(original_size[1], int(original_mid_y + new_height / 2))

    new_mid_x = (x1_new + x2_new) / 2
    new_mid_y = (y1_new + y2_new) / 2
    print(f"New aspect crop midpoint: X={new_mid_x:.2f}, Y={new_mid_y:.2f}")

    return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))

def process_image(file_path, output_dir, confidence_threshold, target_ratios):
    try:
        image = cv2.imread(str(file_path))
        original_image = Image.open(file_path)
        original_size = original_image.size
        print(f"Processing {file_path.name}")
        print(f"Original image size: {original_size}")

        # Perform detection
        results = model(image)[0]

        # Filter detections for person class (class 0 in COCO dataset)
        person_detections = [det for det in results.boxes.data if det[5] == 0 and det[4] >= confidence_threshold]

        if person_detections:
            # Sort detections by confidence and get the highest confidence detection
            best_detection = sorted(person_detections, key=lambda x: x[4], reverse=True)[0]
            x1, y1, x2, y2, conf, class_id = best_detection

            print(f"Detection confidence: {conf:.2f}")
            print(f"Original bounding box: {(x1, y1, x2, y2)}")

            # Calculate aspect ratio
            crop_width, crop_height = x2 - x1, y2 - y1
            detection_aspect_ratio = crop_width / crop_height
            print(f"Detection aspect ratio: {detection_aspect_ratio:.4f}")

            # Get the nearest larger aspect ratio
            nearest_larger_ratio = get_nearest_larger_aspect_ratio(crop_width, crop_height, target_ratios)

            if nearest_larger_ratio:
                print(f"Expanding to ratio: {nearest_larger_ratio[0]}:{nearest_larger_ratio[1]}")
                aspect_bbox = expand_bbox_to_ratio((x1, y1, x2, y2), nearest_larger_ratio, original_size)
                aspect_crop = original_image.crop(aspect_bbox)
                base_name = file_path.stem
                aspect_output_path = output_dir / f"{base_name}_aspectcrop.png"
                aspect_crop.save(aspect_output_path, format='PNG')
                print(f"Saved aspect ratio crop: {aspect_output_path}")
                print(f"Final aspect ratio: {nearest_larger_ratio[0] / nearest_larger_ratio[1]:.4f}")
                print(f"Expanded aspect bbox: {aspect_bbox}")
            else:
                print("No larger standard aspect ratio available. Using original detection crop.")
                aspect_crop = original_image.crop((x1, y1, x2, y2))
                base_name = file_path.stem
                aspect_output_path = output_dir / f"{base_name}_crop.png"
                aspect_crop.save(aspect_output_path, format='PNG')
                print(f"Saved original crop: {aspect_output_path}")

            print(f"Aspect crop size: {aspect_crop.size}")
        else:
            print(f"No person detected in {file_path.name} with confidence above {confidence_threshold}")
        print("--------------------")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == '__main__':
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    for file_path in image_files:
        process_image(file_path, output_dir, confidence_threshold, target_ratios)

    print("All images processed.")
