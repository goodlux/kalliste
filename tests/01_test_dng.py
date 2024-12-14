import os
import pixeltable as pxt

# Initialize Pixeltable
DB_PATH = '/Volumes/g2/kalliste_db'
os.makedirs(DB_PATH, exist_ok=True)
os.environ['PIXELTABLE_DATA_DIR'] = DB_PATH
pxt.init()

# Try to load DNG files from test directory
test_dir = 'img/01_test'
dng_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.dng')]

print(f"Found {len(dng_files)} DNG files in {test_dir}")

try:
    # Attempt to create table and insert DNGs
    try:
        images = pxt.get_table('test_dngs')
        print("Using existing test_dngs table")
    except Exception:
        images = pxt.create_table('test_dngs', {
            'filepath': pxt.String,
            'image': pxt.Image,
            'metadata': pxt.Json
        })
        print("Created new test_dngs table")
    
    # Try to insert each DNG file
    for dng_file in dng_files:
        full_path = os.path.abspath(os.path.join(test_dir, dng_file))
        print(f"\nTrying to insert: {dng_file}")
        try:
            images.insert([{
                'filepath': full_path, 
                'image': full_path
            }])
            print("✓ Successfully inserted")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

except Exception as e:
    print(f"Error setting up table: {str(e)}")

# Try to read back the data
print("\nTrying to read inserted images:")
try:
    results = images.select().collect()
    print("Raw results structure:")
    for i, r in enumerate(results):
        print(f"\nImage {i + 1}:")
        print(type(r))
        print(r)  # Let's see the structure
        if hasattr(r, 'keys'):
            print("Keys:", r.keys())
except Exception as e:
    print(f"Error reading images: {str(e)}")
