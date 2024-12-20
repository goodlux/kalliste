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
    
    # Create output directory if it doesn't exist
    output_dir = Path('/Users/rob/repos/kalliste/test_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = DetectionPipeline(
        model_path='yolov8n.pt',
        detection_config=configs,
        device=None  # Will auto-detect best device
    )
    
    # Process a single image
    image_path = Path('/Users/rob/repos/04_test/IMG_2742.png')
    
    logger.info(f"Processing image: {image_path}")
    results = await pipeline.process_image(image_path, output_dir)
    
    # Print results nicely
    print("\n=== Processing Results ===")
    print(f"Found {len(results)} regions\n")
    
    for i, result in enumerate(results, 1):
        print(f"\nRegion {i}:")
        print(f"Type: {result['region']['type']}")
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