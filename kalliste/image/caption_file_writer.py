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
        "KallistePhotoshootName",
        "KallisteSourceType",
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
        Format: ballerinaLux, {PersonName}, {PhotoshootName}, {Caption}, {OrientationTag}, 
                [aesthetic], [high_quality], [label_{LrLabel}], [LrRating], {LrTags}, 
                [source_{SourceType}], {AdditionalTags}, {Wd14Tags}
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
        
        # Add photoshoot name if it exists
        if "KallistePhotoshootName" in kalliste_tags:
            shoot_name = kalliste_tags["KallistePhotoshootName"].value
            caption_parts.append(shoot_name)
        
        # Add caption if it exists
        if "KallisteCaption" in kalliste_tags:
            caption = kalliste_tags["KallisteCaption"].value
            caption_parts.append(caption)
        
        # Add orientation if it exists
        if "KallisteOrientationTag" in kalliste_tags:
            orientation = kalliste_tags["KallisteOrientationTag"].value
            caption_parts.append(orientation)

        # Add "aesthetic" if KallisteNimaScoreAesthetic is "aesthetic"
        if "KallisteNimaScoreAesthetic" in kalliste_tags:
            nima_aesthetic = kalliste_tags["KallisteNimaScoreAesthetic"].value
            if nima_aesthetic == "aesthetic":
                caption_parts.append("aesthetic")
                logger.debug("Added aesthetic tag based on NIMA score")

        # Add "high_quality" if KallisteNimaScoreTechnical is "high_quality"
        if "KallisteNimaScoreTechnical" in kalliste_tags:
            nima_technical = kalliste_tags["KallisteNimaScoreTechnical"].value
            if nima_technical == "high_quality":
                caption_parts.append("high_quality")
                logger.debug("Added high_quality tag based on NIMA score")

        # Add label_{LrLabel} if it exists
        if "KallisteLrLabel" in kalliste_tags:
            lr_label = kalliste_tags["KallisteLrLabel"].value
            if lr_label:
                label_tag = f"label_{lr_label.lower()}"
                caption_parts.append(label_tag)
                logger.debug(f"Added Lightroom label tag: {label_tag}")

        # Add label_{LrLabel} if it exists
        if "KallisteSourceType" in kalliste_tags:
            lr_label = kalliste_tags["KallisteSourceType"].value
            if lr_label:
                label_tag = f"label_{lr_label.lower()}"
                caption_parts.append(label_tag)
                logger.debug(f"Added Lightroom label tag: {label_tag}")

        # Add LrRating if it exists
        if "KallisteLrRating" in kalliste_tags:
            lr_rating = kalliste_tags["KallisteLrRating"].value
            if lr_rating is not None:
                rating_str = 'unrated' if lr_rating == 0 else f"{lr_rating}_star"
                caption_parts.append(rating_str)
                logger.debug(f"Added Lightroom rating: {rating_str}")
        
        # Add LR tags if they exist
        if "KallisteLrTags" in kalliste_tags:
            lr_tags = kalliste_tags["KallisteLrTags"]
            if isinstance(lr_tags, KallisteBagTag):
                lr_tags_str = ",".join(lr_tags.value)
                caption_parts.append(lr_tags_str)
                logger.debug(f"Added LR tags: {lr_tags_str}")
        
        # Add source type if it exists
        if "KallisteSourceType" in kalliste_tags:
            source_type = kalliste_tags["KallisteSourceType"].value
            caption_parts.append(f"source_{source_type}")
            logger.debug(f"Added source type: {source_type}")
        
        # Add additional tags if they exist
        if "KallisteTags" in kalliste_tags:
            tags = kalliste_tags["KallisteTags"]
            if isinstance(tags, KallisteBagTag):
                tags_str = ",".join(tags.value)
                caption_parts.append(tags_str)
                logger.debug(f"Added additional tags: {tags_str}")
        
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