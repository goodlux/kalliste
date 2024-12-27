"""Image orientation detection using custom model."""
from typing import Dict, List, Union, Optional
from pathlib import Path
import torch
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np

from .base_tagger import BaseTagger
from ..types import TagResult
from ..config import ORIENTATION_MODEL_ID

logger = logging.getLogger(__name__)

class OrientationTagger(BaseTagger):
    """Detects image orientation using a pre-trained model."""
    
    def __init__(self, model=None, processor=None, config: Optional[Dict] = None):
        """Initialize OrientationTagger with model components and config."""
        super().__init__(config=config)
        self.model = model
        self.processor = processor
        
        # Get orientation-specific config
        orientation_config = self.config.get('tagger', {}).get('orientation', {})
        self.confidence_threshold = orientation_config.get('confidence', 0.8)

    def _ensure_valid_image(self, image: Image.Image) -> Image.Image:
        """Ensure image is in a format our model can process."""
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Ensure image is properly loaded
        image.load()
        
        # If image has alpha channel, remove it
        if 'A' in image.getbands():
            # Split into bands and take only RGB
            bands = image.split()
            if len(bands) == 4:  # RGBA
                image = Image.merge('RGB', bands[:3])
        
        return image
        
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate orientation tags for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'orientation' key mapping to list of TagResults,
            sorted by confidence
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not provided. OrientationTagger requires initialized model components.")

        try:
            # Open and validate image
            try:
                image = Image.open(image_path)
                image = self._ensure_valid_image(image)
            except UnidentifiedImageError as e:
                logger.error(f"Could not identify image format for {image_path}")
                raise
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                raise
            
            # Use CPU for MPS device due to known compatibility issues
            device = 'cpu' if self.device == 'mps' else self.device
            
            # Process image with our processor
            try:
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Error during image processing: {e}")
                raise
            
            # Run model inference
            try:
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                raise
            
            # Create results
            results = [
                TagResult(
                    label=label,
                    confidence=float(prob),
                    category='orientation'
                )
                for label, prob in zip(self.model.config.id2label.values(), probs[0])
                if float(prob) >= self.confidence_threshold
            ]
            
            return {'orientation': sorted(results, key=lambda x: x.confidence, reverse=True)}
            
        except Exception as e:
            logger.error(f"Orientation tagging failed for {image_path}: {e}", exc_info=True)
            raise