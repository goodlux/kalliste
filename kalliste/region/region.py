"""Represents a detected region in an image."""
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from PIL import Image
import logging
from ..tag import KallisteStringTag, KallisteBagTag, KallisteIntegerTag, KallisteRealTag

logger = logging.getLogger(__name__)

@dataclass
class Region:
    """Represents a crop region in an image."""
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str  # 'face' or 'person' - keeping this as it's used in code
    confidence: Optional[float] = None  # keeping this as is for detection logic
    rotation: Optional[float] = None
    kalliste_tags: Dict[str, Any] = field(default_factory=dict) # This is the central "kalliste tag registry" for the entire application.

    def __post_init__(self):
        """Initialize kalliste tags based on region data"""
        # Store region_type as a tag as well
        tag = KallisteStringTag("KallisteRegionType", self.region_type)
        self.add_tag(tag)

    def add_tag(self, tag) -> None:
        """Add or update a tag."""
        if tag.name in self.kalliste_tags:
            logger.warning(f"Overwriting existing tag: {tag.name}")
        self.kalliste_tags[tag.name] = tag
    
    def get_tag(self, tag_name: str) -> Optional[Any]:
        """Get a tag if it exists."""
        return self.kalliste_tags.get(tag_name)
    
    
    def get_tag_value(self, tag_name: str, default = None) -> Any:
        """
        Get a tag's value directly, handling different tag types.
        Returns the value in its native type (float for KallisteRealTag, 
        int for KallisteIntegerTag, etc.)
        
        Args:
            tag_name: Name of the tag to retrieve
            default: Optional default value if tag doesn't exist
            
        Returns:
            The tag's value in its native type, or the default value
        """
        tag = self.kalliste_tags.get(tag_name)
        if tag is None:
            return default
            
        return tag.value
    

    def has_tag(self, tag_name: str) -> bool:
        """Check if a tag exists."""
        return tag_name in self.kalliste_tags
    
    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)