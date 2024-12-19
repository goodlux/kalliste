"""Test script for the CropProcessor class."""
from pathlib import Path
from kalliste.processors.crop_processor import CropProcessor
from kalliste.models.exported_image import ExportedImage

def main():
    # Initialize processor
    processor = CropProcessor(model_path='yolov8n.pt', confidence_threshold=0.3)
    
    # Set up paths
    input_dir = Path("/Volumes/m01/kalliste_test/01_test")  # You'll need to create this and add some test images
    output_dir = Path("/Volumes/m01/kalliste_test/01_test_out")
    
    # Process each image
    for image_path in input_dir.glob("*.jpg"):
        print(f"\nProcessing {image_path.name}...")
        
        # Create ExportedImage instance
        image = ExportedImage(image_path)
        
        # Detect and get regions
        regions = processor.process_image(image)
        
        if regions:
            print(f"Found {len(regions)} person(s)")
            for i, region in enumerate(regions):
                print(f"Region {i + 1}:")
                print(f"  Confidence: {region.confidence:.2f}")
                width, height = region.get_dimensions()
                print(f"  Dimensions: {width}x{height}")
                print(f"  Coordinates: ({region.x1}, {region.y1}), ({region.x2}, {region.y2})")
            
            # Save crops
            processor.save_crops(image, regions, output_dir)
            print(f"Saved crops to {output_dir}")
        else:
            print("No persons detected")

if __name__ == "__main__":
    main()