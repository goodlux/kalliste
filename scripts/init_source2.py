# Initialize Pixeltable
pxt.init()

# Drop existing table if it exists
try:
    pxt.drop_table('source_images')
    print("Dropped existing source_images table")
except Exception as e:
    print(f"No existing table to drop: {e}")

# Create new table with our schema
try:
    schema = {
        'project_id': pxt.String,
        'shoot_event': pxt.String,
        'media_type': pxt.String,
        'capture_date': pxt.Int,  # Unix timestamp
        'original_filename': pxt.String,
        'original_path': pxt.String,
        'original_timestamp': pxt.Int,  # Unix timestamp

        # Training text fields
        'auto_caption': pxt.String,
        'lr_keywords': pxt.String,  # Array syntax
        'auto_tags': pxt.String,

        # AI Detection fields
        'pose_tags': pxt.String,
        'clothing_tags': pxt.String,
        'lookalike_tags': pxt.String,
        'detection_tags': pxt.String,

        # Image properties
        'type': pxt.String,
        'orientation': pxt.String,

        # Additional metadata
        'processing_metadata': pxt.Json,
        'created_at': pxt.Int, # Unix timestamp
        'image': pxt.Image
    }

    # Create table
    source_images = pxt.create_table('source_images', schema)
    print("Created source_images table with schema")
except Exception as e:
    print(f"Error creating table: {e}")
    raise e