"""Initialize Pixeltable database with required tables."""
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pixeltable as pxt
from config.config import init_environment

def init_database():
    """Initialize the Pixeltable database with required tables."""
    print("Initializing environment...")
    init_environment()

    print("Initializing Pixeltable...")
    pxt.init()

    # Drop existing table if it exists
    try:
        pxt.drop_table('source_images')
        print("Dropped existing images table")
    except Exception as e:
        print(f"No existing table to drop: {e}")

    # Create new table with our schema
    try:
        schema = {
            # Core identifiers and paths
            'project_id': pxt.String,
            'shoot_event': pxt.String,
            'media_type': pxt.String,  # 'source' or 'crop'
            'capture_date': pxt.Int,  # Unix timestamp
            'original_filename': pxt.String,
            'original_path': pxt.String,
            'original_timestamp': pxt.Int,

            # Training text fields - using Json for flexibility
            'auto_caption': pxt.String,
            'lr_keywords': pxt.Json,  # List of keywords
            'auto_tags': pxt.Json,    # List of auto-generated tags

            # AI Detection fields - using Json for nested structures
            'pose_tags': pxt.Json,      # List of pose descriptions
            'clothing_tags': pxt.Json,   # List of clothing items
            'lookalike_tags': pxt.Json,  # List of similarity matches
            'detection_tags': pxt.Json,  # List of detected objects/features

            # Image properties
            'type': pxt.String,        # 'source', 'person_crop', 'face_crop'
            'orientation': pxt.String,  # 'portrait', 'landscape', 'square'

            # Technical metadata
            'processing_metadata': pxt.Json,  # Processing history, crops, etc
            'created_at': pxt.Int,     # Unix timestamp
            

            
            # The actual image
            'image': pxt.Image
        }

        # Create the table
        images = pxt.create_table('source_images', schema)
        print("Created images table with all columns")

    except Exception as e:
        print(f"Error creating table: {e}")
        raise e

def main():
    """Main function to run database initialization."""
    try:
        init_database()
        print("Database initialization completed successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()