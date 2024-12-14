import os
import pixeltable as pxt
from PIL import Image, ExifTags
import rawpy
from config.settings import TEST_IMAGE_DIR, init_environment

# Initialize environment from config
init_environment()

# Initialize Pixeltable
print("Initializing Pixeltable...")
pxt.init()

# Try to load DNG files from test directory
dng_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith('.dng')]

print(f"Found {len(dng_files)} DNG files in {TEST_IMAGE_DIR}")

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
        full_path = os.path.abspath(os.path.join(TEST_IMAGE_DIR, dng_file))
        print(f"\n{'='*80}\nAnalyzing: {dng_file}\n{'='*80}")
        
        # 1. Examine the raw DNG data
        print("\nExamining raw DNG data...")
        try:
            with rawpy.imread(full_path) as raw:
                print("\nRaw DNG Data")
                print("-" * 12)
                print(f"Raw pattern: {raw.raw_pattern}")
                print(f"Color description: {raw.color_desc}")
                print(f"Raw colors: {raw.raw_colors}")
                print(f"Raw image visible dimensions: {raw.sizes.raw_width} x {raw.sizes.raw_height}")
                print(f"Raw image full dimensions: {raw.sizes.raw_width_full} x {raw.sizes.raw_height_full}")
                
                # Get raw metadata if available
                if hasattr(raw, 'metadata'):
                    print("\nRaw Metadata:")
                    print(raw.metadata)
        except Exception as e:
            print(f"Error reading raw data: {e}")
        
        # 2. Try to insert into Pixeltable
        print("\nInserting into Pixeltable...")
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

# Read and analyze stored images
print("\nAnalyzing stored images:")
try:
    results = images.select().collect()
    for i, r in enumerate(results):
        print(f"\n{'='*80}\nStored Image {i + 1}: {r['filepath']}\n{'='*80}")
        
        # Analyze the stored PIL Image
        if r['image']:
            print("\nStored Image Info")
            print("-" * len("Stored Image Info"))
            print(f"Mode: {r['image'].mode}")
            print(f"Size: {r['image'].size}")
            print(f"Format: {r['image'].format}")
            
            # Try to get EXIF data
            try:
                exif = { ExifTags.TAGS[k]: v for k, v in r['image']._getexif().items() if k in ExifTags.TAGS }
                print("\nEXIF Data:")
                for k, v in exif.items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"No EXIF data available: {e}")
        
        # Print metadata if available
        if r['metadata']:
            print("\nStored Metadata:")
            print(r['metadata'])
        else:
            print("\nNo metadata stored in Pixeltable")
            
except Exception as e:
    print(f"Error analyzing stored images: {str(e)}")
