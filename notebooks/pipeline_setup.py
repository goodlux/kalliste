# %% [markdown]
# # Kalliste Image Import Pipeline
import os
import sys
from pathlib import Path
import pixeltable as pxt
from datetime import datetime
import PIL.Image
from PIL.ExifTags import TAGS
import json

# %% Initialize database and paths
SOURCE_DIR = Path("/Volumes/g2/kalliste_db/test_images/sample_01")
assert SOURCE_DIR.exists(), f"Source directory not found: {SOURCE_DIR}"

pxt.init()
print(f"Connected to Pixeltable...")

# Get reference to source_images table
try:
    source_images = pxt.get_table('source_images')
    print(f"Found source_images table")
except Exception as e:
    print(f"Error accessing table: {e}")
    sys.exit(1)

# %% Process images
# Get list of all images
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.dng']:
    # Print what we're looking for
    print(f"Looking for *{ext} and *{ext.upper()} in {SOURCE_DIR}")
    # Print actual files found for each extension
    files_found = list(SOURCE_DIR.glob(f"*{ext}")) + list(SOURCE_DIR.glob(f"*{ext.upper()}"))
    print(f"Found {len(files_found)} files with extension {ext}:")
    for f in files_found:
        print(f"  - {f}")
    image_files.extend(files_found)

print(f"\nTotal images found: {len(image_files)}")

# Let's also check the directory contents directly
print("\nDirectory contents:")
for f in SOURCE_DIR.iterdir():
    print(f"  - {f.name} ({f.suffix})")

# %% Import images
for image_path in image_files:
    try:
        # Open image and get basic info
        image = PIL.Image.open(image_path)
        stat = image_path.stat()
        
        # Extract EXIF
        exif = {}
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif[tag] = str(value)

        # Collect metadata
        metadata = {
            'filename': image_path.name,
            'original_path': str(image_path.absolute()),
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_size': stat.st_size,
            'exif': exif
        }

        # Prepare record
        record = {
            'project_id': 'sample_01',
            'shoot_event': 'test_import',
            'media_type': 'source',
            'capture_date': int(datetime.fromisoformat(metadata['created']).timestamp()),
            'original_filename': metadata['filename'],
            'original_path': metadata['original_path'],
            'original_timestamp': int(datetime.fromisoformat(metadata['modified']).timestamp()),
            'auto_caption': None,
            'lr_keywords': [],
            'auto_tags': [],
            'pose_tags': [],
            'clothing_tags': [],
            'lookalike_tags': [],
            'detection_tags': [],
            'type': 'source',
            'orientation': 'landscape' if metadata['size'][0] > metadata['size'][1] else 'portrait',
            'processing_metadata': metadata,
            'created_at': int(datetime.now().timestamp()),
            'image': str(image_path)
        }
        
        # Insert into database
        source_images.insert([record])
        print(f"Imported: {image_path.name}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# %% Verify import
# Check total count first
count = source_images.count()
print(f"Total records in table: {count}")

# Try different select variations
print("\nTrying basic select:")
result1 = source_images.select('*')
print(result1)

# Check specific records
print("\nTrying with limit:")
result2 = source_images.select(['original_filename', 'type', 'orientation']).limit(5)
print(result2)

# Try checking for any non-null values using pixeltable expression syntax
print("\nChecking for specific filename:")
result3 = source_images.select(['original_filename']).where(source_images.original_filename != None)
print(result3)

# Let's also try looking at just the filenames
print("\nAll filenames:")
filenames = source_images.select('original_filename')
print(filenames)
# %%
