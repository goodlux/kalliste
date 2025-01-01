"""Represents a detected region in an image."""
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class Region:
    """Represents a crop region in an image."""
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str  # 'face' or 'person'
    confidence: Optional[float] = None
    rotation: Optional[float] = None  # For face regions from LR
    name: Optional[str] = None        # Person name from LR
    tags: Dict[str, str] = field(default_factory=dict)  # key: value pairs

    def add_tag(self, tag_name: str, value: str) -> None:
        """Add a tag with its value."""
        self.tags[tag_name] = value
    
    def get_tag(self, tag_name: str) -> Optional[str]:
        """Get a tag's value if it exists."""
        return self.tags.get(tag_name)

    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    @classmethod
    def from_lightroom_face(cls, metadata: Dict) -> Optional['Region']:
        """Create a Region from Lightroom face metadata.
        
        Args:
            metadata: Dictionary containing Lightroom metadata including:
                - Region Applied To Dimensions W/H: The dimensions of the cropped image
                - Region Area W/H/X/Y: The face region dimensions as percentages
                - Region Rotation: Face region rotation
                - Region Name: Name of the person
        
        Returns:
            Region object if valid face region, None if invalid dimensions
        """
        # Get the cropped image dimensions that the region coordinates are relative to
        try:
            img_width = float(metadata.get('Region Applied To Dimensions W', 0))
            img_height = float(metadata.get('Region Applied To Dimensions H', 0))
            
            # Get region dimensions as percentages of the cropped image
            width_pct = float(metadata.get('Region Area W', 0))
            height_pct = float(metadata.get('Region Area H', 0))
            center_x_pct = float(metadata.get('Region Area X', 0))
            center_y_pct = float(metadata.get('Region Area Y', 0))
            
            # Convert percentages to pixels
            width = width_pct * img_width
            height = height_pct * img_height
            center_x = center_x_pct * img_width
            center_y = center_y_pct * img_height
            
            # Calculate corners from center
            x1 = max(0, int(center_x - width/2))
            y1 = max(0, int(center_y - height/2))
            x2 = min(img_width, int(center_x + width/2))
            y2 = min(img_height, int(center_y + height/2))
            
            region = cls(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                region_type='face',
                rotation=float(metadata.get('Region Rotation', 0)),
                name=metadata.get('Region Name', '')
            )
            
            # Add SDXL tag if dimensions are sufficient
            if x2 - x1 >= 1024 and y2 - y1 >= 1024:
                region.add_tag('SDXL', 'true')
                return region
            
            return None  # Skip regions that are too small
            
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not parse face region metadata: {e}")
            return None