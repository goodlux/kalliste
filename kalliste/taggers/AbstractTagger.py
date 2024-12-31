"""Abstract base class for all Kalliste image taggers."""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from PIL import Image
from ..types import TagResult
from ..model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class AbstractTagger(ABC):
    """Abstract base class that all specific taggers should inherit from."""
    
    def __init__(self, model_id: str, config: Optional[Dict] = None):
        """Initialize tagger.
        
        Args:
            model_id: ID of the model in ModelRegistry to use for tagging
            config: Optional configuration overrides for this tagger.
        """
        # Get model and processor from registry
        model_info = ModelRegistry.get_model(model_id)
        self.model = model_info["model"]
        self.processor = model_info["processor"]

        # Store any additional model components (transform, etc)
        self.model_info = {k: v for k, v in model_info.items() if k != "model"}
        
        # Initialize configuration with defaults
        self.config = self.get_default_config()
        if config:
            # Only update with provided config keys that exist in defaults
            for key in self.config.keys():
                if key in config:
                    self.config[key] = config[key]

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for this tagger."""
        raise NotImplementedError("Taggers must implement get_default_config")

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate tags for an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping categories to lists of TagResults
        """
        with Image.open(image_path) as img:
            return await self.tag_pillow_image(img)
            
    @abstractmethod
    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        """Generate tags for a PIL Image.
        
        Args:
            image: PIL Image to tag
            
        Returns:
            Dictionary mapping categories to lists of TagResults
        """
        pass