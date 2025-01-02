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
        Format: ballerinaLux, {PersonName}, {Caption}, {OrientationTag}, [LR_TagsTBD], {Wd14Tags}
        """
        # Extract values from tag objects, using empty string if tag doesn't exist
        person_name = kalliste_tags.get("KallistePersonName", KallisteStringTag("KallistePersonName", "")).value
        caption = kalliste_tags.get("KallisteCaption", KallisteStringTag("KallisteCaption", "")).value
        orientation = kalliste_tags.get("KallisteOrientationTag", KallisteStringTag("KallisteOrientationTag", "")).value
        
        # For WD14 tags, we need to handle the bag type
        wd14_tag = kalliste_tags.get("KallisteWd14Tags")
        if wd14_tag and isinstance(wd14_tag, KallisteBagTag):
            wd14_tags = ",".join(wd14_tag.value)
        else:
            wd14_tags = ""
        
        # Format caption line
        caption_parts = [
            "ballerinaLux",
            person_name,
            caption,
            orientation,
            "[LR_TagsTBD]",  # Placeholder for now
            wd14_tags
        ]
        
        # Log what we found and didn't find
        for tag in self.REQUIRED_TAGS:
            if tag not in kalliste_tags:
                logger.warning(f"Missing expected tag: {tag}")
        
        return ", ".join(caption_parts)
        
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