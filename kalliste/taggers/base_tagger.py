"""Base class for all Kalliste image taggers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from PIL import Image, UnidentifiedImageError
from ..types import TagResult
from ..model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class BaseTagger(ABC):
    """Base tagger class that all specific taggers should inherit from.
    
    This base class provides:
    - Model initialization from registry
    - Common image loading and validation
    - Configuration management with defaults
    - Standard interface for tagging
    """
    
    def __init__(self, model_id: str, config: Optional[Dict] = None):
        """Initialize base tagger.
        
        Args:
            model_id: ID of the model in ModelRegistry to use for tagging
            config: Optional configuration overrides for this tagger. Will be
                   merged with get_default_config()
        
        Raises:
            KeyError: If model_id not found in registry
            RuntimeError: If models not initialized
        """
        # Get model and associated info from registry
        model_info = ModelRegistry.get_model(model_id)
        self.model = model_info["model"]
        # Store any additional model components (transform, etc)
        self.model_info = {k: v for k, v in model_info.items() if k != "model"}
        
        # Initialize configuration with defaults
        self.config = self.get_default_config()
        if config:
            # Only update with provided config keys that exist in defaults
            # This prevents accidentally adding unsupported config options
            for key in self.config.keys():
                if key in config:
                    self.config[key] = config[key]

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for this tagger.
        
        Returns:
            Dictionary containing default configuration values
            All possible config keys should be present with defaults
        """
        raise NotImplementedError("Taggers must implement get_default_config")

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
            Dictionary mapping categories to lists of TagResults
            Each TagResult should have:
            - label: str
            - confidence: float
            - category: str
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If tagging fails
        """
        pass