"""Image captioning using BLIP2 model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class CaptionTagger(BaseTagger):
    """Generates natural language captions for images using BLIP2 model."""
    
    # Default configuration
    DEFAULT_MAX_LENGTH = 100
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_REPETITION_PENALTY = 1.5
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for BLIP2 caption tagger."""
        return {
            'max_length': cls.DEFAULT_MAX_LENGTH,
            'temperature': cls.DEFAULT_TEMPERATURE,
            'repetition_penalty': cls.DEFAULT_REPETITION_PENALTY
        }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize CaptionTagger.
        
        Args:
            config: Optional configuration dictionary with keys:
                max_length: Maximum caption length in tokens
                    Default: 100
                    Recommended range: 50-200
                temperature: Generation temperature for sampling
                    Default: 1.0
                    Recommended range: 0.1-2.0
                    Higher values make output more diverse but less focused
                repetition_penalty: Penalty for token repetition
                    Default: 1.5
                    Recommended range: 1.0-2.0
                    Higher values reduce word repetition
        """
        super().__init__(model_id="blip2", config=config)

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate a natural language caption for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'caption' key mapping to list containing a single TagResult
            with the generated caption
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If caption generation fails
        """
        try:
            # Use base class method to load and validate image
            image = self._load_and_validate_image(image_path)
            
            # Generate caption
            with torch.inference_mode():
                inputs = self.model.processor(image, return_tensors="pt")
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config['max_length'],
                    do_sample=True,
                    temperature=self.config['temperature'],
                    repetition_penalty=self.config['repetition_penalty']
                )
                
                caption = self.model.processor.decode(output[0], skip_special_tokens=True)
            
            # Return caption with confidence 1.0 since it's a generative model
            return {
                'caption': [
                    TagResult(
                        label=caption.strip(),
                        confidence=1.0,
                        category='caption'
                    )
                ]
            }
            
        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Caption generation failed: {e}")