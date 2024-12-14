from dotenv import load_dotenv
import os
import subprocess

# Load environment variables
load_dotenv()

# Get test directory from .env
test_dir = os.path.join(os.environ['KALLISTE_TEST_IMAGE_ROOT'], '01_test')

# Find all DNG files
dng_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.dng')]

# Dump metadata for each file
for dng_file in sorted(dng_files):
    full_path = os.path.join(test_dir, dng_file)
    output_file = os.path.join(test_dir, dng_file.rsplit('.', 1)[0] + '.txt')

    # Run exiftool and save output
    result = subprocess.run(['exiftool', '-a', '-u', '-g1', full_path],
                          capture_output=True,
                          text=True)

    with open(output_file, 'w') as f:
        f.write(result.stdout)

    print(f"Created {output_file}")
