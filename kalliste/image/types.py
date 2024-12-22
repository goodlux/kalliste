from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
from pathlib import Path

@dataclass
class Detection:
    """Represents a detection result converted from detector Region"""
    type: str  # "face", "person", etc.
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    @classmethod
    def from_region(cls, region) -> 'Detection':
        """Convert a detector Region to a Detection"""
        return cls(
            type=region.region_type,
            bbox=(region.x1, region.y1, region.x2, region.y2),
            confidence=region.confidence if region.confidence else 0.0
        )

@dataclass
class KallisteTag:
    """Single Kalliste tag with metadata"""
    tag: str
    source: str  # e.g. 'lightroom', 'face_detector', 'person_detector'
    confidence: Optional[float] = None

class ProcessingStatus(Enum):
    """Status of image processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"