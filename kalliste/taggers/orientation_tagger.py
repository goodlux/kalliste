"""Image orientation detection using custom model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
import torch

from .base_tagger import BaseTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class OrientationTagger(BaseTagger):
    """Detects image orientation using a pre-trained model.
    
    Predicts image orientation from a set of standard orientations
    (e.g., 'front', 'side', 'back').
    """
    
    # Default configuration
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for orientation detection."""
        return {
            'confidence_threshold': cls.DEFAULT_CONFIDENCE_THRESHOLD
        }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize OrientationTagger.
        
        Args:
            config: Optional configuration dictionary with keys:
                confidence_threshold: Minimum confidence for tag inclusion (0.0-1.0)
                    Default: 0.8
                    Recommended range: 0.5-0.95
                    Lower values will return more possible orientations
        """
        super().__init__(model_id="orientation", config=config)

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate orientation tags for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'orientation' key mapping to list of TagResults,
            sorted by confidence
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If orientation detection fails
        """
        try:
            # Use base class method to load and validate image
            image = self._load_and_validate_image(image_path)
            
            # Run model inference
            with torch.inference_mode():
                outputs = self.model(image)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Create results for predictions above threshold
            results = [
                TagResult(
                    label=label,
                    confidence=float(prob),
                    category='orientation'
                )
                for label, prob in zip(self.model.config.id2label.values(), probs[0])
                if float(prob) >= self.config['confidence_threshold']
            ]
            
            return {'orientation': sorted(results, key=lambda x: x.confidence, reverse=True)}
            
        except Exception as e:
            logger.error(f"Orientation tagging failed for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Orientation detection failed: {e}")