"""Image tagging using WD14 model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch
import numpy as np

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """Tags images using WD14 model for anime/illustration-style tags."""
    
    def __init__(
        self,
        model,
        config: Optional[Dict] = None,
        confidence_threshold: float = 0.35,  # Default confidence threshold
        max_tags: int = 50,                  # Default max tags
        min_tags: int = 5                    # Default min tags
    ):
        """Initialize WD14Tagger.
        
        Args:
            model: Pre-initialized WD14 model
            config: Optional configuration dictionary
            confidence_threshold: Minimum confidence for tag inclusion
                Default: 0.35
                Recommended range: 0.2-0.8
                Lower values return more tags but may be less accurate
            max_tags: Maximum number of tags to return per category
                Default: 50
                Recommended range: 10-100
            min_tags: Minimum number of tags to return (will lower threshold if needed)
                Default: 5
                Recommended range: 1-20
        """
        super().__init__(model, config)
        
        # Get WD14-specific config or use defaults
        wd14_config = self.config.get('wd14', {})
        self.confidence_threshold = wd14_config.get('confidence_threshold', confidence_threshold)
        self.max_tags = wd14_config.get('max_tags', max_tags)
        self.min_tags = wd14_config.get('min_tags', min_tags)
        
        # Ensure min_tags doesn't exceed max_tags
        self.min_tags = min(self.min_tags, self.max_tags)

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate WD14 tags for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary mapping categories to lists of TagResults,
            sorted by confidence. Categories typically include:
            'general', 'character', 'copyright', 'artist'
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If tagging fails
        """
        try:
            # Use base class method to load and validate image
            image = self._load_and_validate_image(image_path)
            
            # Generate tag predictions
            with torch.inference_mode():
                inputs = self.model.processor(image, return_tensors="pt")
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits)[0]
            
            # Convert to numpy for easier processing
            probs = probs.cpu().numpy()
            
            # Get tags above threshold
            tags_above_threshold = [
                (label, float(prob), category)
                for label, prob, category in zip(
                    self.model.config.id2label.values(),
                    probs,
                    self.model.config.id2category.values()
                )
                if prob >= self.confidence_threshold
            ]
            
            # If we have fewer than min_tags, dynamically lower threshold
            if len(tags_above_threshold) < self.min_tags:
                # Sort all predictions by confidence
                all_predictions = [
                    (label, float(prob), category)
                    for label, prob, category in zip(
                        self.model.config.id2label.values(),
                        probs,
                        self.model.config.id2category.values()
                    )
                ]
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                tags_above_threshold = all_predictions[:self.min_tags]
            
            # Group by category
            results: Dict[str, List[TagResult]] = {}
            for label, confidence, category in tags_above_threshold:
                if category not in results:
                    results[category] = []
                
                if len(results[category]) < self.max_tags:
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
                results[category] = results[category][:self.max_tags]
            
            return results
            
        except Exception as e:
            logger.error(f"WD14 tagging failed for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"WD14 tagging failed: {e}")