from dotenv import load_dotenv
import os
import pixeltable as pxt
import json

# Load environment variables
load_dotenv()

# Initialize pixeltable
pxt.init()

# Get our table
images = pxt.get_table('test_dngs')

# For each image in the table
for row in images.select().collect():
    filename = os.path.basename(row['filepath'])
    print(f"\n=== {filename} ===")

    # Look at the table structure
    print("\nTable columns:")
    for key in row.keys():
        print(f"- {key}")

    # Try to access basic metadata
    print("\nBasic metadata:")
    if hasattr(row['image'], 'format'):
        print(f"Format: {row['image'].format}")
    if hasattr(row['image'], 'size'):
        print(f"Size: {row['image'].size}")
    if hasattr(row['image'], 'info'):
        print("Info available:", bool(row['image'].info))

    # Look at what methods are available
    print("\nAvailable methods:")
    methods = [m for m in dir(row['image']) if not m.startswith('_') and callable(getattr(row['image'], m))]
    print(", ".join(methods))
