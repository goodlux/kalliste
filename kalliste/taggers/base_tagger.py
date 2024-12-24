"""Base classes and utilities for Kalliste image tagging system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from pathlib import Path
import torch
import logging
from ..image.types import TagResult

logger = logging.getLogger(__name__)

def get_default_device():
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1).to('mps')
            _ = test_tensor + 1
            logger.info("MPS device validated")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS failed validation: {e}")
            return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class BaseTagger(ABC):
    def __init__(self, device: Optional[str] = None):
        """Initialize tagger with device selection.
        Note: Actual model loading is handled by ModelRegistry."""
        self.device = device or get_default_device()
        logger.info(f"Creating {self.__class__.__name__} instance for device: {self.device}")

    @abstractmethod
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate tags for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping tag categories to lists of TagResults
        """
        pass