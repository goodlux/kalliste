import clipcrop.clipcrop as cc

print("Contents of clipcrop.clipcrop:")
print(dir(cc))

# Try to find any function or class that might be related to cropping
crop_related = [attr for attr in dir(cc) if 'crop' in attr.lower()]
print("\nPotential crop-related attributes:")
for attr in crop_related:
    print(f"{attr}: {getattr(cc, attr)}")

# If no crop-related attributes found, print all callable attributes
if not crop_related:
    print("\nAll callable attributes in clipcrop.clipcrop:")
    for attr in dir(cc):
        if callable(getattr(cc, attr)):
            print(f"{attr}: {getattr(cc, attr)}")