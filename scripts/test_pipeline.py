"""Test script for the detection and tagging pipeline."""
import asyncio
import logging
from pathlib import Path
from kalliste.detectors.base import DetectionConfig
from kalliste.processors.detection_pipeline import DetectionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Configure detection
    configs = [
        DetectionConfig(name='person', confidence_threshold=0.5),
        DetectionConfig(name='face', confidence_threshold=0.5)
    ]
    
    # Set up input and output paths
    input_dir = Path('/Users/rob/repos/kalliste/test_images/01_test/input')
    output_dir = Path('/Users/rob/repos/kalliste/test_images/01_test/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = DetectionPipeline(
        model='yolov8n',
        face_model='yolov8n-face',
        detection_config=configs,
        device=None
    )
    
    # Get all images in input directory
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    # Process each image
    for image_path in image_files:
        logger.info(f"Processing image: {image_path}")
        results = await pipeline.process_image(image_path, output_dir)
        
        # Print results nicely
        print("\n=== Processing Results ===")
        print(f"Found {len(results)} regions\n")
        
        for i, result in enumerate(results, 1):
            print(f"Region {i}:")
            print(f"Type: {result['region']['region_type']}")
            print(f"Confidence: {result['region']['confidence']:.2f}")
            print(f"Coordinates: ({result['region']['x1']}, {result['region']['y1']}) -> "
                  f"({result['region']['x2']}, {result['region']['y2']})")
            
            if 'orientation' in result['tags']:
                orientation = result['tags']['orientation'][0]
                print(f"Orientation: {orientation.label} ({orientation.confidence:.2f})")
            
            if 'caption' in result['tags']:
                print(f"Caption: {result['tags']['caption']}")
            
            if 'wd14' in result['tags']:
                print("Top WD14 tags:")
                for tag in result['tags']['wd14'][:5]:
                    print(f"  - {tag.label} ({tag.confidence:.2f})")
            
            print(f"Crop saved to: {result['crop_path']}")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())