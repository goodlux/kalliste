
# %% [markdown]
# # Kalliste Image Import Pipeline
# Direct path check

from pathlib import Path
import os
SOURCE_DIR = Path("/Volumes/g2/kalliste_db/test_images/sample_01")

# Verify path exists
print(f"Directory exists: {SOURCE_DIR.exists()}")
print(f"Directory is directory: {SOURCE_DIR.is_dir()}")

# List all files, no filtering
print("\nAll files in directory:")
try:
    files = list(SOURCE_DIR.glob("*"))
    for f in files:
        print(f" - {f} (exists: {f.exists()})")
except Exception as e:
    print(f"Error listing directory: {e}")

# Try direct os.listdir() as alternative
print("\nUsing os.listdir:")
try:
    import os
    files = os.listdir(SOURCE_DIR)
    for f in files:
        print(f" - {f}")
except Exception as e:
    print(f"Error with os.listdir: {e}")
# %%
