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
        
        Args:
            source_path: Path to source image
            dest_path: Path to destination image
            kalliste_metadata: Dictionary of Kalliste-specific metadata to add
                Expected keys:
                - photoshoot_id
                - photoshoot_date
                - people
                - caption 
                - wd_tags
                - lr_tags
                - orientation_tag
                - other_tags
                - all_tags
                - source_media
                - crop_type
                - process_version
                - lr_rating
                - lr_label
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Build exiftool command
            cmd = [
                "exiftool",
                "-TagsFromFile", str(source_path),
                "-all:all",                # Copy all metadata
                "-ImageSize=",             # Clear size-related tags
                "-PixelDimensions=",
            ]
            
            # Add Kalliste namespace tags
            for key, value in kalliste_metadata.items():
                if value is not None:
                    # Handle lists by joining with commas
                    if isinstance(value, (list, tuple)):
                        value = ",".join(str(v) for v in value)
                    cmd.extend([f"-XMP-kalliste:{key}={value}"])
            
            # Add destination path
            cmd.append(str(dest_path))
            
            # Run exiftool
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Exiftool error: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy metadata: {e}")
            return False
            
    def extract_kalliste_metadata(self, image_path: Union[str, Path]) -> Optional[Dict]:
        """
        Extract Kalliste-specific metadata from an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Optional[Dict]: Dictionary of Kalliste metadata or None if failed
        """
        try:
            cmd = [
                "exiftool",
                "-json",
                "-XMP-kalliste:all",
                str(image_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Exiftool error: {result.stderr}")
                return None
                
            metadata = json.loads(result.stdout)
            if metadata:
                # Extract just the Kalliste namespace fields
                kalliste_meta = {}
                for key, value in metadata[0].items():
                    if key.startswith("XMP-kalliste:"):
                        clean_key = key.replace("XMP-kalliste:", "")
                        kalliste_meta[clean_key] = value
                return kalliste_meta
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None

    def generate_chroma_metadata(
        self, 
        kalliste_metadata: Dict,
        original_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate metadata structure for ChromaDB storage.
        
        Args:
            kalliste_metadata: Dictionary of Kalliste-specific metadata
            original_metadata: Optional dictionary of original file metadata
            
        Returns:
            Dict: Structured metadata for ChromaDB
        """
        metadata = {
            "kalliste_fast": kalliste_metadata,
        }
        
        if original_metadata:
            metadata["original_metadata"] = original_metadata
            
        return metadata