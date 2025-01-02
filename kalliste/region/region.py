"""Represents a detected region in an image."""
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from PIL import Image
from ..tag import KallisteTag, KallisteTagValue


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
    kalliste_tags: Dict[str, KallisteTagValue] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize kalliste tags based on region data"""
        # Store region_type as a tag as well
        self.add_tag('KallisteRegionType', self.region_type)

    def add_tag(self, tag_name: str, value: KallisteTagValue) -> None:
        """Add a tag with validation"""
        if KallisteTag.validate_value(tag_name, value):
            self.kalliste_tags[tag_name] = value
        else:
            raise ValueError(f"Invalid value type for tag {tag_name}")

    def get_tag(self, tag_name: str) -> Optional[KallisteTagValue]:
        """Get a tag's value if it exists."""
        return self.kalliste_tags.get(tag_name)
    
    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)