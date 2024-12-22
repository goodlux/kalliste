"""Base classes and utilities for Kalliste image tagging system."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Any
from pathlib import Path
from PIL import Image
import logging
import torch

logger = logging.getLogger(__name__)

@dataclass
class TagResult:
    """Stores the results of a single tag/classification"""
    label: str
    confidence: float
    category: str  # e.g., 'orientation', 'style', 'content'

    def __repr__(self):
        return f"{self.category}:{self.label}({self.confidence:.2f})"

def get_default_device():
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1).to('mps')
            _ = test_tensor + 1
            logger.info("MPS device validated")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS failed validation: {e}")
            logger.info("Falling back to CPU")
            return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class BaseTagger:
    """Base class for all image taggers."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the tagger with specified device."""
        self.device = device or get_default_device()
        logger.info(f"Initializing {self.__class__.__name__} on device: {self.device}")
        
        # For models that need CPU when using MPS
        self.model_device = 'cpu' if self.device == 'mps' else self.device

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Tag an image and return results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping tag categories to lists of TagResults
        """
        raise NotImplementedError("Subclasses must implement tag_image")
    
    def _load_model(self):
        """Load the model and any required processors."""
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def _preprocess_image(self, image: Image.Image) -> Any:
        """Preprocess image for model input."""
        raise NotImplementedError("Subclasses must implement _preprocess_image")
    
    def _postprocess_output(self, output: Any) -> Dict[str, List[TagResult]]:
        """Convert model output to TagResults."""
        raise NotImplementedError("Subclasses must implement _postprocess_output")