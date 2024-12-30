"""Image tagging using WD14 model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch
from PIL import Image

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """Tags images using WD14 model for anime/illustration-style tags."""
    
    # Default configuration
    DEFAULT_CONFIDENCE_THRESHOLD = 0.35
    DEFAULT_MAX_TAGS = 50
    DEFAULT_MIN_TAGS = 5
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for WD14 tagger."""
        return {
            'confidence_threshold': cls.DEFAULT_CONFIDENCE_THRESHOLD,
            'max_tags': cls.DEFAULT_MAX_TAGS,
            'min_tags': cls.DEFAULT_MIN_TAGS
        }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize WD14Tagger."""
        super().__init__(model_id="wd14", config=config)
        
        
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate WD14 tags for an image."""
        try:
            # Load and process image
            image = Image.open(image_path)
            # The processor returns {'pixel_values': tensor} but we just want the tensor
            inputs = self.processor(images=image, return_tensors="pt")
            image_tensor = inputs['pixel_values']
            
            # Run inference - pass tensor directly to model
            with torch.inference_mode():
                outputs = self.model(image_tensor)
                probs = torch.sigmoid(outputs)[0]
            # Get id2label and id2category from registry stored values
            id2label = self.model_info["id2label"]
            id2category = self.model_info["id2category"]
            
            # Get tags above threshold
            tags_above_threshold = [
                (label, float(prob), category)
                for label, prob, category in zip(
                    id2label.values(),
                    probs.cpu(),
                    id2category.values()
                )
                if float(prob) >= self.config['confidence_threshold']
            ]
            
            # If we have fewer than min_tags, dynamically lower threshold
            if len(tags_above_threshold) < self.config['min_tags']:
                all_predictions = [
                    (label, float(prob), category)
                    for label, prob, category in zip(
                        id2label.values(),
                        probs.cpu(),
                        id2category.values()
                    )
                ]
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                tags_above_threshold = all_predictions[:self.config['min_tags']]
            
            # Group by category
            results: Dict[str, List[TagResult]] = {}
            for label, confidence, category in tags_above_threshold:
                if category not in results:
                    results[category] = []
                
                if len(results[category]) < self.config['max_tags']:
                    results[category].append(
                        TagResult(
                            label=label,
                            confidence=confidence,
                            category=category
                        )
                    )
            
            # Sort each category by confidence
            for category in results:
                results[category].sort(key=lambda x: x.confidence, reverse=True)
                results[category] = results[category][:self.config['max_tags']]
            
            return results
            
        except Exception as e:
            logger.error(f"WD14 tagging failed for {image_path}: {e}")
            raise RuntimeError(f"WD14 tagging failed: {e}")