"""BLIP2-based image captioning tagger."""

from typing import Dict, List, Union, Any
from pathlib import Path
from PIL import Image
import torch
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .base_tagger import BaseTagger, TagResult
from ..config import BLIP2_MODEL_ID

logger = logging.getLogger(__name__)

class CaptionTagger(BaseTagger):
    """Image captioning using BLIP2 model."""
    
    def __init__(self, model_id: str = BLIP2_MODEL_ID, **kwargs):
        """Initialize BLIP2 caption tagger.
        
        Args:
            model_id: HuggingFace model ID for BLIP2
            **kwargs: Additional arguments passed to BaseTagger
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load BLIP2 model and processor."""
        logger.info(f"Loading BLIP2 model: {self.model_id}")
        try:
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            
            # Always load BLIP2 on CPU when using MPS
            device = "cpu" if self.device == "mps" else self.device
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                load_in_8bit=True if device == "cuda" else False
            ).to(device)
            
            logger.info(f"BLIP2 model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP2 model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Any:
        """Preprocess image for BLIP2 model."""
        device = "cpu" if self.device == "mps" else self.device
        return self.processor(image, return_tensors="pt").to(device)
    
    def _postprocess_output(self, output: Any) -> Dict[str, List[TagResult]]:
        """Convert BLIP2 output to TagResults."""
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return {
            'caption': [
                TagResult(
                    label=caption.strip(),
                    confidence=1.0,  # BLIP2 doesn't provide confidence scores
                    category='caption'
                )
            ]
        }
    
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate caption for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'caption' key mapping to list of TagResults
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self._preprocess_image(image)
            
            # Move model to CPU if using MPS
            if self.device == "mps":
                self.model = self.model.to("cpu")
            
            # Generate caption
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=1,
                length_penalty=1,
                repetition_penalty=1.5
            )
            
            # Move model back to MPS if needed
            if self.device == "mps":
                self.model = self.model.to("mps")
            
            # Process results
            return self._postprocess_output(output)
            
        except Exception as e:
            logger.error(f"BLIP2 captioning failed: {e}")
            return {
                'caption': [
                    TagResult(
                        label="Failed to generate caption",
                        confidence=0.0,
                        category='caption'
                    )
                ]
            }