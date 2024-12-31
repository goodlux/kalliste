"""Image orientation detection using Hugging Face model."""
from typing import Dict, List, Optional
from PIL import Image
import logging
import torch

from .AbstractTagger import AbstractTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class OrientationTagger(AbstractTagger):
    """Detects image orientation (front, side, back)."""
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize OrientationTagger."""
        super().__init__(model_id="orientation", config=config)

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            'confidence_threshold': cls.DEFAULT_CONFIDENCE_THRESHOLD
        }
    
    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        """Generate orientation tags for a PIL Image."""
        try:
            logger.debug("Preparing image for orientation processor")
            inputs = self.processor(images=image, return_tensors="pt")
            
            logger.debug(f"Moving inputs to device: {self.model.device}")
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }
            
            logger.debug("Running orientation detection")
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                logger.debug(f"Raw probabilities: {probs}")
            
            # Create results for ALL orientations, not just those above threshold
            results = [
                TagResult(
                    label=f"{self.model.config.id2label[i]}_view",  # Add _view suffix
                    confidence=float(prob),
                    category='orientation'
                )
                for i, prob in enumerate(probs)
            ]
            
            final_results = {'orientation': sorted(results, key=lambda x: x.confidence, reverse=True)}
            logger.debug(f"Returning orientation results: {final_results}")
            return final_results
            
        except Exception as e:
            logger.error(f"Orientation detection failed: {str(e)}", exc_info=True)
            return {'orientation': []}