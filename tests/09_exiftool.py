#!/usr/bin/env python3
from pathlib import Path
import subprocess

# Set your input directory here
INPUT_DIR = "/Volumes/g2/kalliste_db/test_images/01_test"

def process_dng(dng_file: Path):
    """Extract full image, edited preview and metadata from a DNG file"""
    base_name = dng_file.stem

    # Output files
    full_path = dng_file.parent / f"{base_name}_full.jpg"     # Full resolution
    edited_path = dng_file.parent / f"{base_name}_edited.jpg"  # JpgFromRaw (edited version)
    metadata_path = dng_file.parent / f"{base_name}.txt"

    print(f"\nProcessing: {dng_file.name}")

    # Extract full resolution image
    full_cmd = ['exiftool', '-b', '-SubIFD', str(dng_file)]
    try:
        with open(full_path, 'wb') as f:
            subprocess.run(full_cmd, stdout=f, check=True)
        print(f"Saved full resolution image to: {full_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting full resolution: {e}")

    # Extract JpgFromRaw (edited version)
    edited_cmd = ['exiftool', '-b', '-JpgFromRaw', str(dng_file)]
    try:
        with open(edited_path, 'wb') as f:
            subprocess.run(edited_cmd, stdout=f, check=True)
        print(f"Saved edited preview to: {edited_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting edited preview: {e}")

    # Extract metadata
    metadata_cmd = ['exiftool', '-a', '-u', '-g1', str(dng_file)]
    try:
        with open(metadata_path, 'w') as f:
            subprocess.run(metadata_cmd, stdout=f, text=True, check=True)
        print(f"Saved metadata to: {metadata_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting metadata: {e}")

def main():
    input_dir = Path(INPUT_DIR)

    # Find all DNG files
    dng_files = list(input_dir.glob('*.dng')) + list(input_dir.glob('*.DNG'))
    print(f"Found {len(dng_files)} DNG files in {input_dir}")

    # Process each file
    for dng_file in dng_files:
        process_dng(dng_file)

if __name__ == "__main__":
    main()
