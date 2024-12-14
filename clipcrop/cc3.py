import os
from pathlib import Path
from clipcrop.clipcrop import ClipCrop
from PIL import Image

# Adjustable parameters
min_score = 0.7  # Minimum confidence score for accepting a detection
prompt = "woman wearing a swimsuit"

# Input and output directory paths
input_dir = Path("/Volumes/p02/2024/2024_MaFiLux_lines_s4_shoot_2_video_2")
output_dir = Path("/Volumes/p02/2024/2024_MaFiLux_lines_s4_shoot_2_video_2_crop")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# Initialize variables for ClipCrop models
DFE, DM, CLIPM, CLIPP = None, None, None, None

for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = input_dir / filename

        try:
            # Open the original image
            original_image = Image.open(input_path)
            original_size = original_image.size
            print(f"Processing {filename}")
            print(f"Original image size: {original_size}")

            # Initialize ClipCrop for the current image
            cc = ClipCrop(str(input_path))

            # Load models if not already loaded
            if DFE is None:
                DFE, DM, CLIPM, CLIPP = cc.load_models()

            # Extract image using ClipCrop with the specified prompt
            result = cc.extract_image(DFE, DM, CLIPM, CLIPP, prompt, num=1)

            # Process result
            if result and isinstance(result[0], dict) and 'image' in result[0]:
                detection_score = result[0].get('score', 0)
                print(f"Detection score: {detection_score}")

                if detection_score >= min_score:
                    clipcrop_image = result[0]['image']

                    # Save the ClipCrop crop
                    base_name = Path(filename).stem
                    crop_output_path = output_dir / f"{base_name}_crop.png"
                    clipcrop_image.save(crop_output_path)
                    print(f"Saved crop: {crop_output_path}")
                    print(f"Crop size: {clipcrop_image.size}")
                else:
                    print(f"Detection score below threshold. No crop saved.")
            else:
                print(f"No subject found in {filename}. No crop saved.")

            print("--------------------")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All images processed.")
