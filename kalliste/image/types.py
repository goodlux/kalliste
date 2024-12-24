from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TagResult:
    """Represents a single tag result from a tagger"""
    label: str
    confidence: float
    category: str  # e.g., 'orientation', 'style', 'content'
    
    def __post_init__(self):
        """Validate tag result attributes after initialization"""
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            logger.error(f"Invalid confidence value: {self.confidence}")
            raise ValueError("Confidence must be between 0 and 1")

    def __repr__(self):
        return f"{self.category}:{self.label}({self.confidence:.2f})"

@dataclass
class Detection:
    """Represents a detection result converted from detector Region"""
    type: str  # "face", "person", etc.
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    def __post_init__(self):
        """Validate detection attributes after initialization"""
        # Validate bbox coordinates
        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            logger.error(f"Invalid bounding box coordinates: {self.bbox}")
            raise ValueError("Invalid bounding box: coordinates out of order")
        if x1 < 0 or y1 < 0:
            logger.warning(f"Bounding box has negative coordinates: {self.bbox}")
            
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            logger.error(f"Invalid confidence value: {self.confidence}")
            raise ValueError("Confidence must be between 0 and 1")
            
        # Log successful creation
        logger.debug(f"Created Detection: type={self.type}, bbox={self.bbox}, confidence={self.confidence:.3f}")
    
    @classmethod
    def from_region(cls, region) -> 'Detection':
        """Convert a detector Region to a Detection"""
        logger.debug(f"Converting region to Detection: {region}")
        try:
            detection = cls(
                type=region.region_type,
                bbox=(region.x1, region.y1, region.x2, region.y2),
                confidence=region.confidence if region.confidence else 0.0
            )
            logger.debug(f"Successfully converted region to Detection: {detection}")
            return detection
        except AttributeError as e:
            logger.error(f"Failed to convert region - missing attribute: {e}", exc_info=True)
            raise ValueError(f"Invalid region format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error converting region: {e}", exc_info=True)
            raise

@dataclass
class KallisteTag:
    """Single Kalliste tag with metadata"""
    tag: str
    source: str  # e.g. 'lightroom', 'face_detector', 'person_detector'
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate tag attributes after initialization"""
        # Validate tag format
        if not self.tag or not isinstance(self.tag, str):
            logger.error(f"Invalid tag value: {self.tag}")
            raise ValueError("Tag must be a non-empty string")
            
        # Validate source
        if not self.source or not isinstance(self.source, str):
            logger.error(f"Invalid source value: {self.source}")
            raise ValueError("Source must be a non-empty string")
            
        # Validate confidence if present
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
                logger.error(f"Invalid confidence value: {self.confidence}")
                raise ValueError("Confidence must be between 0 and 1")
                
        # Log successful creation - fixed the f-string formatting issue
        conf_str = f"{self.confidence:.3f}" if self.confidence is not None else "None"
        logger.debug(
            f"Created KallisteTag: tag='{self.tag}', source='{self.source}', "
            f"confidence={conf_str}"
        )

    def to_dict(self) -> dict:
        """Convert tag to dictionary format"""
        logger.debug(f"Converting KallisteTag to dict: {self.tag}")
        return {
            'tag': self.tag,
            'source': self.source,
            'confidence': self.confidence
        }

class ProcessingStatus(Enum):
    """Status of image processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    
    def __str__(self):
        return self.value
        
    @classmethod
    def transition_allowed(cls, current: 'ProcessingStatus', new: 'ProcessingStatus') -> bool:
        """Check if status transition is allowed"""
        # Define allowed transitions
        allowed_transitions = {
            cls.PENDING: {cls.PROCESSING, cls.ERROR},
            cls.PROCESSING: {cls.COMPLETE, cls.ERROR},
            cls.COMPLETE: {cls.ERROR},  # Only allow transition to ERROR from COMPLETE
            cls.ERROR: set()  # No transitions allowed from ERROR
        }
        
        is_allowed = new in allowed_transitions[current]
        if not is_allowed:
            logger.warning(
                f"Invalid status transition attempted: {current} -> {new}. "
                f"Allowed transitions from {current}: {allowed_transitions[current]}"
            )
        else:
            logger.debug(f"Status transition: {current} -> {new}")
            
        return is_allowed
        
    @classmethod
    def from_string(cls, status_str: str) -> 'ProcessingStatus':
        """Convert string to ProcessingStatus, with validation"""
        try:
            return cls(status_str.lower())
        except ValueError:
            logger.error(f"Invalid status string: {status_str}")
            valid_values = [s.value for s in cls]
            raise ValueError(f"Invalid status. Must be one of: {valid_values}")

# Add status transition validation function
def validate_status_transition(current: ProcessingStatus, new: ProcessingStatus):
    """Validate status transition and log appropriately"""
    if not ProcessingStatus.transition_allowed(current, new):
        error_msg = f"Invalid status transition: {current} -> {new}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Valid status transition: {current} -> {new}")
