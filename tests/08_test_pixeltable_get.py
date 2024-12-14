from dotenv import load_dotenv
import os
import pixeltable as pxt
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize pixeltable
pxt.init()

# Create directory for previews
preview_dir = os.path.join(os.environ['KALLISTE_TEST_IMAGE_ROOT'], 'previews')
os.makedirs(preview_dir, exist_ok=True)

# Get our table
images = pxt.get_table('test_dngs')

# For each image in the table
for row in images.select().collect():
    filename = os.path.basename(row['filepath'])
    print(f"\n=== {filename} ===")

    img = row['image']

    # Try different ways to get previews
    print("\nTrying to get previews:")

    # Method 1: Try to get child images after reopening
    try:
        img_copy = Image.open(io.BytesIO(img.tobytes()))
        children = img_copy.get_child_images()
        print(f"Found {len(children)} child images")

        # Save each preview
        for i, child in enumerate(children):
            preview_path = os.path.join(preview_dir, f"{filename}_preview_{i}.jpg")
            child.save(preview_path)
            print(f"Saved preview {i} to {preview_path}")
            print(f"Preview size: {child.size}")
            print(f"Preview mode: {child.mode}")
    except Exception as e:
        print(f"Error with method 1: {e}")

    # Method 2: Try to access any preview data directly
    try:
        if hasattr(img, 'preview'):
            preview_path = os.path.join(preview_dir, f"{filename}_direct_preview.jpg")
            img.preview.save(preview_path)
            print(f"Saved direct preview to {preview_path}")
    except Exception as e:
        print(f"Error with method 2: {e}")
