"""Initialize Pixeltable database with required tables and functions."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pixeltable as pxt
from config.config import PIXELTABLE_DATA, init_environment

def init_database():
    """Initialize the Pixeltable database with required tables."""
    # Initialize environment variables and directories
    init_environment()

    # Initialize Pixeltable connection
    pxt.init()
    print(f"Initialized Pixeltable database at {PIXELTABLE_DATA}")

    # Drop existing table if it exists
    try:
        pxt.drop_table('images')
        print("Dropped existing images table")
    except Exception as e:
        print(f"No existing table to drop: {e}")

    # Create new table with our schema
    try:
        images = pxt.create_table('images', {
            'image': pxt.Image,           # The image itself
            'project_id': pxt.String,     # e.g., 'balletLux'
            'shoot_event': pxt.String,    # e.g., 'lines_s4'
            'media_type': pxt.String,     # 'photo' or 'video_frame'
            'capture_date': pxt.Date,
            'original_filename': pxt.String,
            'original_path': pxt.String,
            'original_timestamp': pxt.Timestamp,

            # Training text fields
            'auto_caption': pxt.String,
            'lr_keywords': pxt.Array(pxt.String),
            'auto_tags': pxt.Array(pxt.String),

            # AI Detection fields
            'pose_tags': pxt.Array(pxt.String),
            'clothing_tags': pxt.Array(pxt.String),
            'lookalike_tags': pxt.Array(pxt.String),
            'detection_tags': pxt.Array(pxt.String),

            # Image properties
            'type': pxt.String,          # 'person_crop', 'face_crop'
            'orientation': pxt.String,    # 'portrait', 'landscape', 'square'

            # Additional metadata
            'processing_metadata': pxt.Json,
            'created_at': pxt.Timestamp
        })
        print("Created images table with all columns")

        # Create indexes
        # Note: Pixeltable might handle some indexing automatically
        # You may need to add manual index creation if needed

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
