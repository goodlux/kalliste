"""Test script for reading Adobe DNG metadata."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import exifread
from PIL import Image
import subprocess
from config.config import TEST_IMAGE_DIR

def get_exiftool_metadata(file_path):
    """Extract metadata using exiftool if available."""
    try:
        result = subprocess.run(['exiftool', str(file_path)], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            metadata = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            return metadata
    except FileNotFoundError:
        print("exiftool not found - skipping exiftool metadata")
    return None

def get_exifread_metadata(file_path):
    """Extract metadata using exifread."""
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
            return {tag: str(tags[tag]) for tag in tags.keys()}
    except Exception as e:
        print(f"Error reading exif data: {e}")
        return None

def get_pillow_metadata(file_path):
    """Extract metadata using Pillow."""
    try:
        with Image.open(file_path) as img:
            return img.info
    except Exception as e:
        print(f"Error reading Pillow metadata: {e}")
        return None

def print_metadata(file_name, exiftool_data, exif_data, pillow_data):
    """Print metadata in an organized format."""
    print(f"\nProcessing: {file_name}")
    print("="*50)

    if exiftool_data:
        print("\n=== ExifTool Metadata (Selected Fields) ===")
        interesting_fields = [
            'DNG Version', 'Camera Model', 'Preview Image', 
            'Rating', 'Label', 'Keywords', 'Caption-Abstract',
            'Create Date', 'Image Size', 'Color Space'
        ]
        for field in interesting_fields:
            if field in exiftool_data:
                print(f"{field:20}: {exiftool_data[field]}")

    if exif_data:
        print("\n=== EXIF Metadata (Selected Fields) ===")
        interesting_tags = [
            'Image Model', 'Image DateTime', 'Image Software',
            'EXIF DateTimeOriginal', 'EXIF ExposureTime',
            'EXIF FNumber', 'EXIF ISOSpeedRatings'
        ]
        for tag in interesting_tags:
            if tag in exif_data:
                print(f"{tag:20}: {exif_data[tag]}")

    if pillow_data:
        print("\n=== Pillow Metadata ===")
        for key, value in pillow_data.items():
            print(f"{key:20}: {value}")

def main():
    """Process all DNG files in the test directory."""
    test_dir = Path(TEST_IMAGE_DIR)
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    # Find all DNG files (case insensitive)
    dng_files = []
    for ext in ['*.dng', '*.DNG']:
        dng_files.extend(list(test_dir.glob(ext)))
    print(f"\nFound {len(dng_files)} DNG files in {test_dir}")
    
    # Process each file
    for dng_file in dng_files:
        exiftool_data = get_exiftool_metadata(dng_file)
        exif_data = get_exifread_metadata(dng_file)
        pillow_data = get_pillow_metadata(dng_file)
        print_metadata(dng_file.name, exiftool_data, exif_data, pillow_data)

if __name__ == "__main__":
    main()
