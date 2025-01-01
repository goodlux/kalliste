"""Metadata processing and copying for Kalliste images."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)

class MetadataProcessor:
    """Handles metadata operations for Kalliste images."""
    
    def __init__(self):
        """Initialize the metadata processor."""
        pass
        
    def copy_metadata(
        self,
        source_path: Union[str, Path],
        dest_path: Union[str, Path],
        kalliste_metadata: Dict[str, any]
    ) -> bool:
        """
        Copy metadata from source to destination image and add Kalliste-specific tags.
        Also writes a companion .txt file with tag information.
        """
        try:
            logger.error("Starting metadata copy")
            
            # Get region object from metadata
            region = kalliste_metadata.get('region')
            if not region:
                logger.error("No region provided in metadata")
                return False

            logger.error(f"Got region with tags: {region.tags}")
            
            # Write text file first
            txt_path = str(dest_path).rsplit('.', 1)[0] + '.txt'
            logger.error(f"Will write text file to: {txt_path}")
            
            # Format text file content using internal tag names
            content = [
                "ballerinaLux",
                region.get_tag("PersonName", ""),
                region.get_tag("Caption", ""),
                region.get_tag("OrientationTag", ""),
                "[LR_TagsTBD]",
                region.get_tag("Wd14Tags", "")
            ]
            
            txt_content = ", ".join(content)
            logger.error(f"Text content will be: {txt_content}")
            
            # Write text file
            with open(txt_path, 'w') as f:
                f.write(txt_content)
            logger.error(f"Wrote text file to {txt_path}")

            # Build exiftool command
            cmd = [
                "exiftool",
                "-TagsFromFile", str(source_path),
                "-all:all",
            ]
            
            # Add Kalliste XMP tags from region tags with Kalliste prefix
            tag_mapping = {
                "PersonName": "KallistePersonName",
                "Caption": "KallisteCaption",
                "OrientationTag": "KallisteOrientationTag",
                "Wd14Tags": "KallisteWd14Tags",
            }
            
            for tag_name, value in region.tags.items():
                if tag_name in tag_mapping:
                    xmp_tag = tag_mapping[tag_name]
                    cmd.extend([f"-XMP-Kalliste:{xmp_tag}={value}"])
            
            # Add destination path and overwrite flag
            cmd.extend([str(dest_path), "-overwrite_original"])
            
            # Log full command for debugging
            logger.error(f"Full exiftool command: {' '.join(cmd)}")
            
            # Run exiftool
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Exiftool error: {result.stderr}")
                return False
            
            logger.error(f"Exiftool stdout: {result.stdout}")
            logger.error(f"Exiftool stderr: {result.stderr}")
                    
            return True
                
        except Exception as e:
            logger.error(f"Failed to process metadata: {str(e)}")
            return False