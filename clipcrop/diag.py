import clipcrop

print("Clipcrop attributes and methods:")
print(dir(clipcrop))

print("\nClipcrop file location:")
print(clipcrop.__file__)

print("\nClipcrop version:")
print(clipcrop.__version__ if hasattr(clipcrop, '__version__') else "Version not available")

# Try to find any function or class that might be related to cropping
crop_related = [attr for attr in dir(clipcrop) if 'crop' in attr.lower()]
print("\nPotential crop-related attributes:")
for attr in crop_related:
    print(f"{attr}: {getattr(clipcrop, attr)}")

# If no crop-related attributes found, print all callable attributes
if not crop_related:
    print("\nAll callable attributes in clipcrop:")
    for attr in dir(clipcrop):
        if callable(getattr(clipcrop, attr)):
            print(f"{attr}: {getattr(clipcrop, attr)}")