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
            config_path = Path(__file__).parent.parent.parent / "config" / "exiftool" / "kalliste.config"
            
            # Build exiftool command
            cmd = [
                "exiftool",
                "-config", str(config_path),
                "-TagsFromFile", str(source_path),
                "-all:all",
                "-ImageSize=",
                "-PixelDimensions=",
            ]
            
            # Map our internal metadata to XMP-Kalliste namespace tags
            if "tags" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:Tags=" + ",".join(kalliste_metadata["tags"])])
                
            if "tag_confidences" in kalliste_metadata:
                confidences = [str(conf) for conf in kalliste_metadata["tag_confidences"].values()]
                cmd.extend(["-XMP-Kalliste:TagConfidences=" + ",".join(confidences)])
                
            if "tag_sources" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:TagSources=" + ",".join(kalliste_metadata["tag_sources"].values())])
                
            if "region_type" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:DetectionType=" + kalliste_metadata["region_type"]])
                
            if "confidence" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:DetectionConfidence=" + str(kalliste_metadata["confidence"])])
                
            if "process_version" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:ProcessingVersion=" + kalliste_metadata["process_version"]])
                
            if "original_path" in kalliste_metadata:
                cmd.extend(["-XMP-Kalliste:OriginalPath=" + str(kalliste_metadata["original_path"])])
            
            # Add destination path
            cmd.append(str(dest_path))
            cmd.append("-overwrite_original")
            
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