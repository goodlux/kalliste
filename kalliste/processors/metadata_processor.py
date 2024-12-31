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
        try:
            # Get path to our kalliste config
            config_path = Path(__file__).parent.parent.parent / "config" / "exiftool" / "kalliste.config"

            # First try just writing a test tag
            logger.info("Attempting to write test tag...")
            test_cmd = [
                "exiftool",
                "-config", str(config_path),
                "-v",
                "-kalliste:TestTag=test",
                str(dest_path)
            ]
            
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True
            )
            
            logger.info("Test tag write output:")
            logger.info(f"STDOUT: {result.stdout}")
            logger.info(f"STDERR: {result.stderr}")

            # Try reading all kalliste tags
            logger.info("\nAttempting to read kalliste tags...")
            read_cmd = [
                "exiftool",
                "-config", str(config_path),
                "-v",
                "-kalliste:all",
                str(dest_path)
            ]
            
            result = subprocess.run(
                read_cmd,
                capture_output=True,
                text=True
            )
            
            logger.info("Tag read output:")
            logger.info(f"STDOUT: {result.stdout}")
            logger.info(f"STDERR: {result.stderr}")

            # Now proceed with original metadata copying
            logger.info("\nProceeding with full metadata copy...")
            cmd = [
                "exiftool",
                "-config", str(config_path),
                "-v",
                "-TagsFromFile", str(source_path),
                "-all:all",
                "-ImageSize=",
                "-PixelDimensions=",
            ]
            
            # Add Kalliste namespace tags
            for key, value in kalliste_metadata.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        value = ",".join(str(v) for v in value)
                    cmd.extend([f"-kalliste:{key}={value}"])

            # TEST CODE - Remove after verifying tag writing works
            cmd.extend([
                "-kalliste:TestTag=TestTag",
                "-kalliste:TestTagConfidence=0.95"
            ])
            
            # Add destination path
            cmd.append(str(dest_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            logger.info("Full metadata copy output:")
            logger.info(f"STDOUT: {result.stdout}")
            logger.info(f"STDERR: {result.stderr}")
            
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