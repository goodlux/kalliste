"""Base classes and utilities for detection framework."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import numpy as np

@dataclass
class Region:
    """A detected region in an image."""
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str  # e.g., 'person', 'face', 'cat'
    confidence: Optional[float] = None
    rotation: Optional[float] = None
    name: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def get_dimensions(self) -> Tuple[int, int]:
        """Get width and height of region."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    def get_area(self) -> int:
        """Get area of region in pixels."""
        width, height = self.get_dimensions()
        return width * height
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (width/height) of region."""
        width, height = self.get_dimensions()
        return width / height
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of region."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def expand_to_ratio(self, target_ratio: float, max_dims: Tuple[int, int]) -> 'Region':
        """Expand region to match target aspect ratio while maintaining center."""
        current_width, current_height = self.get_dimensions()
        center_x, center_y = self.get_center()
        max_width, max_height = max_dims
        
        current_ratio = current_width / current_height
        
        if current_ratio < target_ratio:
            # Need to increase width
            new_width = int(current_height * target_ratio)
            new_height = current_height
        else:
            # Need to increase height
            new_width = current_width
            new_height = int(current_width / target_ratio)
            
        # Calculate new coordinates
        half_width = new_width // 2
        half_height = new_height // 2
        
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(max_width, center_x + half_width)
        y2 = min(max_height, center_y + half_height)
        
        return Region(
            x1=x1, y1=y1, x2=x2, y2=y2,
            region_type=self.region_type,
            confidence=self.confidence,
            rotation=self.rotation,
            name=self.name,
            tags=self.tags.copy()
        )

    def add_tag(self, tag: str) -> None:
        """Add a tag to the region."""
        self.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the region."""
        self.tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if region has a specific tag."""
        return tag in self.tags

@dataclass
class DetectionConfig:
    """Configuration for a detection type."""
    name: str  # e.g., 'person', 'face', 'cat'
    confidence_threshold: float
    min_size: Optional[Tuple[int, int]] = None  # Minimum size for SDXL
    preferred_aspect_ratios: Optional[List[Tuple[int, int]]] = None  # e.g. [(1,1), (4,3)]
    
    def get_ratio_targets(self) -> List[float]:
        """Convert preferred aspect ratios to float values."""
        if not self.preferred_aspect_ratios:
            return [1.0]  # Default to square if no preferences
        return [w/h for w, h in self.preferred_aspect_ratios]

class BaseDetector(ABC):
    """Base class for all detectors."""
    
    def __init__(self, config: List[DetectionConfig]):
        self.config = {det.name: det for det in config}
    
    @abstractmethod
    def detect(self, image_path: Path) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected regions
        """
        pass
    
    def find_best_ratio(self, current_ratio: float, config: DetectionConfig) -> float:
        """Find the closest preferred aspect ratio for a region."""
        targets = config.get_ratio_targets()
        return min(targets, key=lambda x: abs(x - current_ratio))
    
    def adjust_for_sdxl(self, region: Region, image_size: Tuple[int, int]) -> Optional[Region]:
        """Adjust region to meet SDXL requirements.
        
        Args:
            region: Region to adjust
            image_size: (width, height) of original image
            
        Returns:
            Adjusted region or None if requirements can't be met
        """
        config = self.config[region.region_type]
        width, height = region.get_dimensions()
        
        # Check minimum size if specified
        if config.min_size and (width < config.min_size[0] or height < config.min_size[1]):
            return None
        
        # Adjust aspect ratio if preferred ratios are specified
        if config.preferred_aspect_ratios:
            current_ratio = region.get_aspect_ratio()
            target_ratio = self.find_best_ratio(current_ratio, config)
            region = region.expand_to_ratio(target_ratio, image_size)
        
        return region