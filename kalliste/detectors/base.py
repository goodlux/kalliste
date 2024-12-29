"""Base classes and utilities for detection framework."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Region:
    """A detected region in an image."""
    x1: int
    y1: int 
    x2: int
    y2: int
    region_type: str
    region_index: int
    confidence: float
    tags: Set[str] = field(default_factory=set)

class BaseDetector(ABC):
    """Base class for all detectors."""
    
    def __init__(self, config: dict):
        """Initialize detector with configuration."""
        self.config = config
        self.model = None  # Will be set by child classes via ModelRegistry
    
    def _validate_image_path(self, image_path: Path) -> None:
        """Validate image path exists."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    @abstractmethod
    def detect(self, image_path: Path, **kwargs) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            **kwargs: Additional detector-specific parameters
            
        Returns:
            List[Region]: List of detected regions with type and index
        """
        pass