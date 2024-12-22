"""Subject orientation classification. Returns BACK, FRONT, SIDE based on subject orientation to camera."""

from typing import Dict, List, Union, Any
from pathlib import Path
from PIL import Image
import torch
import logging
from transformers import AutoModelForImageClassification, AutoProcessor, pipeline

from .base_tagger import BaseTagger, TagResult
from ..config import ORIENTATION_MODEL_ID

logger = logging.getLogger(__name__)

class OrientationTagger(BaseTagger):
    """Image orientation classification."""
    
    def __init__(self, model_id: str = ORIENTATION_MODEL_ID, **kwargs):
        """Initialize orientation tagger.
        
        Args:
            model_id: HuggingFace model ID for orientation classifier
            **kwargs: Additional arguments passed to BaseTagger
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load orientation classifier model and processor."""
        try:
            logger.info(f"Loading orientation model: {self.model_id}")
            # Use CPU for MPS, otherwise use specified device
            device = "cpu" if self.device == "mps" else self.device
            
            # Load model and processor
            model = AutoModelForImageClassification.from_pretrained(
                self.model_id
            ).to(device)
            
            processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Create pipeline
            self.pipeline = pipeline(
                "image-classification",
                model=model,
                image_processor=processor,
                device=device
            )
            
            logger.info(f"Orientation model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load orientation model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for orientation classifier."""
        return image  # Pipeline handles preprocessing
    
    def _postprocess_output(self, output: List[Dict[str, float]]) -> Dict[str, List[TagResult]]:
        """Convert pipeline output to TagResults."""
        return {
            'orientation': [
                TagResult(
                    label=result['label'],
                    confidence=result['score'],
                    category='orientation'
                )
                for result in output
            ]
        }
    
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Classify image orientation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'orientation' key mapping to list of TagResults
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and classify image
            image = Image.open(image_path).convert('RGB')
            results = self.pipeline(image)
            
            # Process results
            return self._postprocess_output(results)
            
        except Exception as e:
            logger.error(f"Orientation classification failed: {e}")
            return {
                'orientation': [
                    TagResult(
                        label="unknown",
                        confidence=0.0,
                        category='orientation'
                    )
                ]
            }