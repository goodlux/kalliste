"""Represents an image exported from Lightroom with its metadata."""
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
    tags: Set[str] = field(default_factory=set)

    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the region."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the region."""
        self.tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if region has a specific tag."""
        return tag in self.tags

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
                region.add_tag('SDXL')
                return region
            
            return None  # Skip regions that are too small
            
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not parse face region metadata: {e}")
            return None


class ExportedImage:
    """Represents an image exported from Lightroom with its metadata."""
    
    def __init__(self, path: Path):
        self.source_path: Path = path
        self.lr_metadata: Dict = {}  # Will store Lightroom metadata
        self.face_regions: List[Region] = []
        
    def extract_metadata(self) -> None:
        """Extract metadata from image file."""
        with Image.open(self.source_path) as img:
            self.image_size = img.size
            # TODO: Extract actual metadata using exiftool or similar
            # For now, this is a placeholder
    
    def load_face_regions(self, metadata: Dict) -> None:
        """Extract face regions from Lightroom metadata.
        
        Expected metadata format:
        {
            'Region Applied To Dimensions W': width of cropped image,
            'Region Applied To Dimensions H': height of cropped image,
            'Region Area H': height_percentage in cropped image,
            'Region Area W': width_percentage in cropped image,
            'Region Area X': center_x_percentage in cropped image,
            'Region Area Y': center_y_percentage in cropped image,
            'Region Rotation': rotation,
            'Region Name': name,
            'Region Type': 'Face'
        }
        """
        # For each face region in metadata
        if metadata.get('Region Type') == 'Face':
            region = Region.from_lightroom_face(metadata)
            if region is not None:
                self.face_regions.append(region)
    
    def get_all_regions(self) -> List[Region]:
        """Get all regions (both faces and persons)."""
        return self.face_regions  # Will add person regions later when detected