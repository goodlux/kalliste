"""Represents an image exported from Lightroom with its metadata."""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
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

    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    def is_valid_sdxl_size(self) -> bool:
        """Check if region meets SDXL size requirements."""
        width, height = self.get_dimensions()
        if self.region_type == 'face':
            return width >= 1024 and height >= 1024
        return True  # For person crops, we'll adjust to valid ratios

    @classmethod
    def from_lightroom_face(cls, metadata: Dict, image_size: Tuple[int, int]) -> 'Region':
        """Create a Region from Lightroom face metadata."""
        img_width, img_height = image_size
        
        # Convert percentages to pixels
        width = float(metadata.get('Region Area W', 0)) * img_width
        height = float(metadata.get('Region Area H', 0)) * img_height
        center_x = float(metadata.get('Region Area X', 0)) * img_width
        center_y = float(metadata.get('Region Area Y', 0)) * img_height
        
        # Calculate corners from center
        x1 = int(center_x - width/2)
        y1 = int(center_y - height/2)
        x2 = int(center_x + width/2)
        y2 = int(center_y + height/2)
        
        return cls(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            region_type='face',
            rotation=float(metadata.get('Region Rotation', 0)),
            name=metadata.get('Region Name', '')
        )


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
            'Person In Image': 'Name',
            'Region Applied To Dimensions W': width,
            'Region Applied To Dimensions H': height,
            'Region Rotation': rotation,
            'Region Name': name,
            'Region Type': 'Face',
            'Region Area H': height_percent,
            'Region Area W': width_percent,
            'Region Area X': center_x_percent,
            'Region Area Y': center_y_percent
        }
        """
        if not hasattr(self, 'image_size'):
            self.extract_metadata()
        
        # For each face region in metadata
        if metadata.get('Region Type') == 'Face':
            region = Region.from_lightroom_face(metadata, self.image_size)
            if region.is_valid_sdxl_size():
                self.face_regions.append(region)
    
    def get_all_regions(self) -> List[Region]:
        """Get all regions (both faces and persons)."""
        return self.face_regions  # Will add person regions later when detected