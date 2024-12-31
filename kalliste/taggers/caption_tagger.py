"""Image captioning using BLIP2 model."""
from typing import Dict, List, Optional
from PIL import Image
import logging
import torch

from .AbstractTagger import AbstractTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class CaptionTagger(AbstractTagger):
    """Generates natural language captions using BLIP2."""
    
    DEFAULT_MAX_LENGTH = 100
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_REPETITION_PENALTY = 1.5

    def __init__(self, config: Optional[Dict] = None):
        """Initialize CaptionTagger."""
        super().__init__(model_id="blip2", config=config)
    
    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            'max_length': cls.DEFAULT_MAX_LENGTH,
            'temperature': cls.DEFAULT_TEMPERATURE,
            'repetition_penalty': cls.DEFAULT_REPETITION_PENALTY
        }

    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        """Generate a caption for a PIL Image."""
        try:
            logger.debug("Preparing image for BLIP2 processor")
            inputs = self.processor(image, return_tensors="pt")
            
            logger.debug(f"Moving inputs to device: {self.model.device}")
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            logger.debug("Generating caption")
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config['max_length'],
                    do_sample=True,
                    temperature=self.config['temperature'],
                    repetition_penalty=self.config['repetition_penalty']
                )
                
                caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
                logger.debug(f"Generated caption: {caption}")
            
                # Always create a tag result with the caption
                result = {
                    'caption': [
                        TagResult(
                            label=caption,
                            confidence=1.0,  # We could adjust this based on model confidence if needed
                            category='caption'
                        )
                    ]
                }
                logger.debug(f"Returning caption result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}", exc_info=True)
            return {'caption': []}