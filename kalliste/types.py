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
