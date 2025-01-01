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
        """
        try:
            config_path = "/Users/rob/repos/kalliste/config/exiftool/kalliste.config"
            
            # Build exiftool command with just source metadata copy and our test tag
            cmd = [
                "exiftool",
                "-config", str(config_path),
                "-TagsFromFile", str(source_path),
                "-all:all",
                "-XMP-kalliste:kallisteTest1=DummyValue1",
                "-XMP-kalliste:kallisteTest2=DummyValue2",
                "-XMP-kalliste:kallisteTest3=DummyValue3",
                "-XMP-kalliste:kallisteTest4=DummyValue4",
            ]
            
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
            logger.error(f"Failed to copy metadata: {str(e)}", exc_info=True)
            return False