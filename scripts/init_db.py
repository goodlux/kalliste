"""Initialize Pixeltable database with required tables."""
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pixeltable as pxt
from config.config import init_environment

def init_database():
    """Initialize the Pixeltable database with required tables."""
    # Initialize environment variables and directories first
    print("Initializing environment...")
    init_environment()

    print("Initializing Pixeltable...")
    # Initialize Pixeltable
    pxt.init()

    # Drop existing table if it exists
    try:
        pxt.drop_table('images')
        print("Dropped existing images table")
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

        # Create base table first
        images = pxt.create_table('images', schema)
        print("Created base table")

        # Add image column after table creation
         # images.add_column(pxt.Image, 'image')
        print("Added image column")

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
