"""Test script for the CropProcessor class."""
from pathlib import Path
from kalliste.processors.crop_processor import CropProcessor
from kalliste.models.exported_image import ExportedImage

def main():
    # Initialize processor
    processor = CropProcessor(model_path='yolov8n.pt', confidence_threshold=0.4)  # Increased confidence
    
    # Set up paths
    input_dir = Path("/Volumes/m01/kalliste_test/02_test")
    output_dir = Path("/Volumes/m01/kalliste_test/02_test_out")
    
    # Include both jpg and png
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    # Process each image
    for image_path in sorted(image_files):
        print(f"\nProcessing {image_path.name}...")
        
        # Create ExportedImage instance
        image = ExportedImage(image_path)
        
        # Example face metadata - you'll need to extract this from your files
        face_metadata = {
            'Person In Image': 'ChRoLux',
            'Region Applied To Dimensions W': '3456',
            'Region Applied To Dimensions H': '5184',
            'Region Rotation': '-1.70045',
            'Region Name': 'ChRoLux',
            'Region Type': 'Face',
            'Region Area H': '0.04541',
            'Region Area W': '0.06808',
            'Region Area X': '0.50681',
            'Region Area Y': '0.35321'
        }
        
        # Load face regions if available
        try:
            image.load_face_regions(face_metadata)
            if image.face_regions:
                print(f"Found {len(image.face_regions)} valid face region(s)")
        except Exception as e:
            print(f"Error loading face regions: {e}")
        
        # Detect and get person regions
        person_regions = processor.process_image(image)
        
        # Process person detections
        if person_regions:
            print(f"Found {len(person_regions)} person(s)")
            for i, region in enumerate(person_regions):
                print(f"Person region {i + 1}:")
                print(f"  Confidence: {region.confidence:.2f}")
                width, height = region.get_dimensions()
                print(f"  Dimensions: {width}x{height}")
        
        # Combine all regions and save crops
        all_regions = person_regions + image.face_regions
        if all_regions:
            processor.save_crops(image, all_regions, output_dir)
            print(f"Saved all crops to {output_dir}")
        else:
            print("No regions detected")

if __name__ == "__main__":
    main()