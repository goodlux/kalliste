import os
import pixeltable as pxt

# Hardcode paths for now just to test
PIXELTABLE_DATA = '/Volumes/g2/kalliste_db/pixeltable/data'
POSTGRES_DATA = '/Volumes/g2/kalliste_db/pixeltable/postgres'
TEST_IMAGE_DIR = '/Volumes/g2/kalliste_db/test_images/01_test'

# Set up pixeltable
os.environ['PIXELTABLE_DATA_DIR'] = PIXELTABLE_DATA
os.environ['PGDATA'] = POSTGRES_DATA

# Initialize Pixeltable
print("Initializing Pixeltable...")
pxt.init()

# Get DNG files
dng_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith('.dng')]
print(f"Found {len(dng_files)} DNG files")

# Create or get table
try:
    images = pxt.get_table('test_dngs')
    print("Using existing test_dngs table")
except:
    images = pxt.create_table('test_dngs', {
        'filepath': pxt.String,
        'image': pxt.Image,
        'metadata': pxt.Json
    })
    print("Created new test_dngs table")

# Insert images
for dng_file in dng_files:
    full_path = os.path.join(TEST_IMAGE_DIR, dng_file)
    print(f"\nProcessing: {dng_file}")
    images.insert([{'filepath': full_path, 'image': full_path}])
    print("✓ Inserted")
