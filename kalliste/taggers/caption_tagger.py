"""Image captioning using BLIP2 model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import torch
import logging
from PIL import Image

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class CaptionTagger(BaseTagger):
    """Generates image captions using BLIP2 model.
    
    Note: Model initialization is handled by ModelRegistry. The tagger expects
    to receive initialized model and processor instances.
    """
    
    def __init__(self, model=None, processor=None, config: Optional[Dict] = None):
        """Initialize CaptionTagger with model components and config."""
        super().__init__(config=config)
        self.model = model
        self.processor = processor
        
        # Get caption-specific config
        caption_config = self.config.get('tagger', {}).get('caption', {})
        self.max_length = caption_config.get('max_length', 100)
        self.temperature = caption_config.get('temperature', 1.0)
        self.repetition_penalty = caption_config.get('repetition_penalty', 1.5)

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate a caption for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'caption' key mapping to list containing a single TagResult
            with the generated caption
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not provided. CaptionTagger requires initialized model components.")

        try:
            image = Image.open(image_path).convert('RGB')
            # Use CPU for MPS device due to known compatibility issues
            device = "cpu" if self.device == "mps" else self.device
            inputs = self.processor(image, return_tensors="pt").to(device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    do_sample=True,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty
                )
            
            caption = self.processor.decode(output[0], skip_special_tokens=True)
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
            raise