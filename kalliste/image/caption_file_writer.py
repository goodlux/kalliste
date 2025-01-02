"""Writes kalliste tags to caption text file."""
from pathlib import Path
from typing import Dict, Any
import logging
from ..tag import KallisteStringTag, KallisteBagTag

logger = logging.getLogger(__name__)

class CaptionFileWriter:
    """Writes kalliste tags to caption text file."""
    
    # Tags we're looking for
    REQUIRED_TAGS = [
        "KallistePersonName",
        "KallisteCaption",
        "KallisteOrientationTag",
        "KallisteWd14Tags"
    ]
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        
    def write_caption(self, kalliste_tags: Dict[str, Any]) -> bool:
        """
        Main function to write caption file.
        Steps:
        1. Extract required tags
        2. Format caption text
        3. Write to file
        """
        try:
            if not self._validate_path():
                return False
                
            caption_text = self._format_caption(kalliste_tags)
            return self._write_to_file(caption_text)
            
        except Exception as e:
            logger.error(f"Failed to write caption file: {e}")
            return False
            
    def _validate_path(self) -> bool:
        """Ensure output path's parent directory exists."""
        try:
            output_dir = self.output_path.parent
            if not output_dir.exists():
                logger.error(f"Output directory does not exist: {output_dir}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating output path: {e}")
            return False
        
    def _format_caption(self, kalliste_tags: Dict[str, Any]) -> str:
        """
        Format caption text from kalliste tags.
        Format: ballerinaLux, {PersonName}, {Caption}, {OrientationTag}, {LrTags}, {Wd14Tags}
        """
        # Start with mandatory prefix
        caption_parts = ["ballerinaLux"]
        
        # Add person name if it exists
        if "KallistePersonName" in kalliste_tags:
            name = kalliste_tags["KallistePersonName"].value
            logger.debug(f"Found person name: {name}")
            caption_parts.append(name)
        else:
            logger.warning("No KallistePersonName tag found")
        
        # Add caption if it exists
        if "KallisteCaption" in kalliste_tags:
            caption = kalliste_tags["KallisteCaption"].value
            caption_parts.append(caption)
        
        # Add orientation if it exists
        if "KallisteOrientationTag" in kalliste_tags:
            orientation = kalliste_tags["KallisteOrientationTag"].value
            caption_parts.append(orientation)
        
        # Add LR tags if they exist
        if "KallisteLrTags" in kalliste_tags:
            lr_tags = kalliste_tags["KallisteLrTags"]
            if isinstance(lr_tags, KallisteBagTag):
                lr_tags_str = ",".join(lr_tags.value)
                caption_parts.append(lr_tags_str)
                logger.debug(f"Added LR tags: {lr_tags_str}")
        
        # Add WD14 tags if they exist
        if "KallisteWd14Tags" in kalliste_tags:
            wd14_tags = kalliste_tags["KallisteWd14Tags"]
            if isinstance(wd14_tags, KallisteBagTag):
                wd14_tags_str = ",".join(wd14_tags.value)
                caption_parts.append(wd14_tags_str)
        
        # Join parts with commas, filtering out any empty strings
        result = ", ".join(filter(None, caption_parts))
        logger.debug(f"Generated caption: {result}")
        return result
            
    def _write_to_file(self, caption_text: str) -> bool:
        """Write formatted caption to file."""
        try:
            with open(self.output_path, 'w') as f:
                f.write(caption_text)
            logger.info(f"Successfully wrote caption to: {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing caption file: {e}")
            return False