"""Base class for all Kalliste image taggers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from PIL import Image, UnidentifiedImageError
from ..types import TagResult


logger = logging.getLogger(__name__)

class BaseTagger(ABC):
    """Base tagger class that all specific taggers should inherit from."""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """Initialize base tagger.
        
        Args:
            model: Pre-initialized model (should already have device configured)
            config: Optional configuration overrides
        """
        self.model = model
        self.config = config or {}

    def _load_and_validate_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and validate image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL.Image: Loaded and validated image in RGB format
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            UnidentifiedImageError: If image format cannot be identified
            ValueError: If image validation fails
        """
        # Convert to Path object if string
        if isinstance(image_path, str):
            image_path = Path(image_path)
            
        # Check file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Ensure image is properly loaded
            image.load()
            
            # Remove alpha channel if present
            if 'A' in image.getbands():
                bands = image.split()
                if len(bands) == 4:  # RGBA
                    image = Image.merge('RGB', bands[:3])
                    
            return image
            
        except UnidentifiedImageError:
            logger.error(f"Could not identify image format: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise ValueError(f"Image validation failed: {e}")

    @abstractmethod
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate tags for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping tag categories to lists of TagResults
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If tagging fails
        """
        pass