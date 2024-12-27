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
    
    def __init__(
        self,
        model,
        config: Optional[Dict] = None,
        max_length: int = 100,          # Default max tokens
        temperature: float = 1.0,        # Default temperature
        repetition_penalty: float = 1.5  # Default repetition penalty
    ):
        """Initialize CaptionTagger.
        
        Args:
            model: Pre-initialized BLIP2 model
            config: Optional configuration dictionary
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
        super().__init__(model, config)
        
        # Get caption-specific config or use defaults
        caption_config = self.config.get('caption', {})
        self.max_length = caption_config.get('max_length', max_length)
        self.temperature = caption_config.get('temperature', temperature)
        self.repetition_penalty = caption_config.get('repetition_penalty', repetition_penalty)

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
                    max_new_tokens=self.max_length,
                    do_sample=True,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty
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