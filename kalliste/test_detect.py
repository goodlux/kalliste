"""Test script for the detection framework."""
from pathlib import Path
import argparse
from typing import List

from PIL import Image, ImageDraw

from kalliste.processors.crop_processor import CropProcessor
from kalliste.models.exported_image import ExportedImage
from config.config import YOLO_WEIGHTS

def setup_processor(person_model_path: str, face_model_path: str) -> CropProcessor:
    """Configure and initialize the processor."""
    return CropProcessor(
        person_model_path=person_model_path,
        face_model_path=face_model_path,
        confidence_threshold=0.4
    )

def save_debug_image(image_path: Path, regions: List[ExportedImage], output_path: Path) -> None:
    """Save a debug image showing all detected regions."""
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # Draw each region with different colors per type
        colors = {
            'person': (255, 0, 0),  # Red
            'face': (0, 255, 0),    # Green
            'cat': (0, 0, 255)      # Blue
        }
        
        for region in regions:
            color = colors.get(region.region_type, (255, 255, 0))  # Yellow for unknown types
            
            # Draw rectangle
            draw.rectangle([region.x1, region.y1, region.x2, region.y2], 
                         outline=color, width=3)
            
            # Draw label with confidence
            label = f"{region.region_type}"
            if region.confidence is not None:
                label += f" {region.confidence:.2f}"
            if 'SDXL' in region.tags:
                label += " SDXL"
            
            draw.text((region.x1, region.y1 - 10), label, fill=color)
        
        # Save debug image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"Saved debug image to {output_path}")

def process_image(image_path: Path, processor: CropProcessor, output_dir: Path, debug: bool = False) -> None:
    """Process a single image through the detection pipeline."""
    print(f"\nProcessing {image_path.name}...")
    
    # Create ExportedImage instance
    image = ExportedImage(image_path)
    
    # Process image to get all regions
    regions = processor.process_image(image)
    
    if regions:
        print(f"\nFound {len(regions)} regions:")
        for i, region in enumerate(regions, 1):
            width, height = region.get_dimensions()
            confidence_str = f"{region.confidence:.2f}" if region.confidence is not None else "N/A"
            print(f"Region {i}: {region.region_type}")
            print(f"  Confidence: {confidence_str}")
            print(f"  Dimensions: {width}x{height}")
            print(f"  Tags: {', '.join(region.tags)}")
        
        # Save crops
        processor.save_crops(image, regions, output_dir)
        
        # Save debug image if requested
        if debug:
            debug_path = output_dir / 'debug' / f"{image_path.stem}_debug.png"
            save_debug_image(image_path, regions, debug_path)
    else:
        print("No regions detected")

def main():
    parser = argparse.ArgumentParser(description='Test detection framework')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for crops')
    parser.add_argument('--person-model', type=str, 
                       default=str(YOLO_WEIGHTS['object']),
                       help='Path to YOLO person detection model')
    parser.add_argument('--face-model', type=str,
                       default=str(YOLO_WEIGHTS['face']),
                       help='Path to YOLO face detection model')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug images showing detections')
    args = parser.parse_args()
    
    # Convert paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = setup_processor(args.person_model, args.face_model)
    
    # Get list of images to process
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    # Process each image
    for image_path in sorted(image_files):
        try:
            process_image(image_path, processor, output_dir, args.debug)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()