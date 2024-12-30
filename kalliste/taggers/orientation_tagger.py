"""Image orientation detection using Hugging Face model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch
from PIL import Image

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class OrientationTagger(BaseTagger):
    """Detects image orientation (front, side, back)."""
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize OrientationTagger."""
        # Pass model_id to parent class
        super().__init__(model_id="orientation", config=config)



    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            'confidence_threshold': cls.DEFAULT_CONFIDENCE_THRESHOLD
        }
    

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate orientation tags for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to model device if needed
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            results = [
                TagResult(
                    label=self.model.config.id2label[i],
                    confidence=float(prob),
                    category='orientation'
                )
                for i, prob in enumerate(probs)
                if float(prob) >= self.config['confidence_threshold']
            ]
            
            return {'orientation': sorted(results, key=lambda x: x.confidence, reverse=True)}
            
        except Exception as e:
            logger.error(f"Orientation tagging failed for {image_path}: {e}")
            raise RuntimeError(f"Orientation detection failed: {e}")