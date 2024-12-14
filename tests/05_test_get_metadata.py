from dotenv import load_dotenv
import os
import pixeltable as pxt
import json

# Load environment variables
load_dotenv()

# Initialize pixeltable
pxt.init()

# Get the table and show what's in it
images = pxt.get_table('test_dngs')
print("\nContents of test_dngs table:")
print("----------------------------")

for row in images.select().collect():
    print(f"\nFilename: {os.path.basename(row['filepath'])}")
    if row['metadata']:
        print("Metadata:")
        print(json.dumps(row['metadata'], indent=2))
    else:
        print("No metadata found")
