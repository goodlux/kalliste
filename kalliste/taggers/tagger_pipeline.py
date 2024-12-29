"""Image tagging using WD14 model."""
from typing import Dict, List, Union, Optional, Set
from pathlib import Path
import logging
import torch
import numpy as np

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """Tags images using WD14 model for anime/illustration-style tags."""
    
    # Default configuration
    DEFAULT_CONFIDENCE_THRESHOLD = 0.35
    DEFAULT_MAX_TAGS = 50
    DEFAULT_MIN_TAGS = 5
    DEFAULT_BLACKLIST: Set[str] = set()  # Empty set by default
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for WD14 tagger."""
        return {
            'confidence': cls.DEFAULT_CONFIDENCE_THRESHOLD,  # Changed to match config.yaml
            'max_tags': cls.DEFAULT_MAX_TAGS,
            'min_tags': cls.DEFAULT_MIN_TAGS,
            'blacklist': list(cls.DEFAULT_BLACKLIST)  # Convert set to list for config
        }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize WD14Tagger.
        
        Args:
            config: Optional configuration dictionary with keys:
                confidence: Minimum confidence for tag inclusion
                    Default: 0.35
                    Recommended range: 0.2-0.8
                max_tags: Maximum number of tags per category
                    Default: 50
                    Recommended range: 10-100
                min_tags: Minimum number of tags (will lower threshold if needed)
                    Default: 5
                    Recommended range: 1-20
                blacklist: List of tags to exclude from results
                    Default: []
        """
        super().__init__(model_id="wd14", config=config)
        
        # Convert blacklist to set for efficient lookups
        self.blacklist = set(self.config.get('blacklist', []))

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
                inputs = self.transform(image).unsqueeze(0)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs.logits)[0]
            
            # Convert to numpy for easier processing
            probs = probs.cpu().numpy()
            
            # Filter out blacklisted tags and get tags above threshold
            tags_above_threshold = [
                (label, float(prob), category)
                for label, prob, category in zip(
                    self.model.config.id2label.values(),
                    probs,
                    self.model.config.id2category.values()
                )
                if prob >= self.config['confidence']  # Changed to match config.yaml
                and label.lower() not in self.blacklist  # Add blacklist filtering
            ]
            
            # If we have fewer than min_tags, dynamically lower threshold
            # but still respect blacklist
            if len(tags_above_threshold) < self.config['min_tags']:
                # Sort all predictions by confidence
                all_predictions = [
                    (label, float(prob), category)
                    for label, prob, category in zip(
                        self.model.config.id2label.values(),
                        probs,
                        self.model.config.id2category.values()
                    )
                    if label.lower() not in self.blacklist  # Respect blacklist even when lowering threshold
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
            logger.error(f"WD14 tagging failed for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"WD14 tagging failed: {e}")