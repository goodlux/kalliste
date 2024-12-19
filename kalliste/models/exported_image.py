"""Represents an image exported from Lightroom with its metadata."""
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Region:
    """Represents a crop region in an image."""
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str  # 'face' or 'person'
    confidence: Optional[float] = None

    def get_dimensions(self) -> tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    def is_valid_sdxl_size(self) -> bool:
        """Check if region meets SDXL size requirements."""
        width, height = self.get_dimensions()
        if self.region_type == 'face':
            return width >= 1024 and height >= 1024
        return True  # For person crops, we'll adjust to valid ratios


class ExportedImage:
    """Represents an image exported from Lightroom with its metadata."""
    
    def __init__(self, path: Path):
        self.source_path: Path = path
        self.lr_metadata: Dict = {}  # Will store Lightroom metadata
        self.face_regions: List[Region] = []  # Will store face regions from LR
        
    def load_face_regions(self) -> None:
        """Extract face regions from Lightroom metadata."""
        # TODO: Implement face region extraction from LR metadata
        pass
        
    def extract_metadata(self) -> None:
        """Extract relevant metadata from the image."""
        # TODO: Implement metadata extraction
        pass