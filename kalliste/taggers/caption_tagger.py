"""Image captioning using BLIP2 model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch
from PIL import Image

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class CaptionTagger(BaseTagger):
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

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate a caption for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to model device if needed
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config['max_length'],
                    do_sample=True,
                    temperature=self.config['temperature'],
                    repetition_penalty=self.config['repetition_penalty']
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
            logger.error(f"Caption generation failed for {image_path}: {e}")
            raise RuntimeError(f"Caption generation failed: {e}")