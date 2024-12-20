"""
Image tagging and classification system for Kalliste.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, AutoModelForImageClassification, AutoProcessor
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import logging
import numpy as np
from PIL import Image
import os

logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

def get_default_device():
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        return "mps"  # Use Metal Performance Shaders on Mac
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

@dataclass
class TagResult:
    """Stores the results of a single tag/classification"""
    label: str
    confidence: float
    category: str  # e.g., 'orientation', 'style', 'content'

    def __repr__(self):
        return f"{self.category}:{self.label}({self.confidence:.2f})"

class ImageTagger:
    """
    Manages multiple image classification and tagging models.
    """
    
    MODEL_CONFIGS = {
        'orientation': {
            'model_id': "LucyintheSky/pose-estimation-front-side-back",
            'cache_dir': WEIGHTS_DIR / "orientation",
            'model_type': "vit"
        },
        'wd14': {
            'model_id': "SmilingWolf/wd-vit-large-tagger-v3",
            'cache_dir': WEIGHTS_DIR / "wd14",
            'model_type': "vit"
        },
        'blip2': {
            'model_id': "Salesforce/blip2-opt-2.7b",
            'cache_dir': WEIGHTS_DIR / "blip2"
        }
    }
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ImageTagger with specified device.
        
        Args:
            device: Optional device specification ('mps', 'cuda', 'cpu', or None)
                   If None, will automatically select best available device
        """
        self.device = device or get_default_device()
        logger.info(f"Initializing ImageTagger on device: {self.device}")
        
        # Create weights directory if it doesn't exist
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        for config in self.MODEL_CONFIGS.values():
            config['cache_dir'].mkdir(parents=True, exist_ok=True)
            
        # Pre-download models to cache
        self._ensure_models_downloaded()
        
        self.classifiers: Dict[str, Any] = {}
        self.threshold = 0.35  # Confidence threshold for WD14 tags
        self._initialize_classifiers()

    def _ensure_models_downloaded(self):
        """Ensure all models are downloaded to cache"""
        logger.info("Ensuring all models are downloaded...")
        for name, config in self.MODEL_CONFIGS.items():
            logger.info(f"Downloading {name} model...")
            try:
                snapshot_download(
                    repo_id=config['model_id'],
                    local_dir=config['cache_dir'],
                    local_dir_use_symlinks=False,
                    allow_patterns=["*.safetensors", "*.json"]  # Only download safetensors and config files
                )
            except Exception as e:
                logger.error(f"Error downloading {name} model: {e}")
                raise

    def _initialize_classifiers(self):
        """Initialize all classifiers"""
        logger.info("Loading classifiers...")
        try:
            device = self.device if self.device != "mps" else "cpu"
            
            # Orientation classifier
            logger.info(f"Loading orientation classifier from {self.MODEL_CONFIGS['orientation']['cache_dir']}")
            orientation_model = AutoModelForImageClassification.from_pretrained(
                self.MODEL_CONFIGS['orientation']['model_id'],
                cache_dir=self.MODEL_CONFIGS['orientation']['cache_dir']
            )
            orientation_processor = AutoProcessor.from_pretrained(
                self.MODEL_CONFIGS['orientation']['model_id'],
                cache_dir=self.MODEL_CONFIGS['orientation']['cache_dir']
            )
            self.classifiers['orientation'] = pipeline(
                "image-classification",
                model=orientation_model,
                image_processor=orientation_processor,
                device=device
            )

            # WD14 Tagger
            logger.info(f"Loading WD14 tagger from {self.MODEL_CONFIGS['wd14']['cache_dir']}")
            wd14_model = AutoModelForImageClassification.from_pretrained(
                self.MODEL_CONFIGS['wd14']['model_id'],
                cache_dir=self.MODEL_CONFIGS['wd14']['cache_dir']
            )
            wd14_processor = AutoProcessor.from_pretrained(
                self.MODEL_CONFIGS['wd14']['model_id'],
                cache_dir=self.MODEL_CONFIGS['wd14']['cache_dir']
            )
            self.classifiers['wd14'] = pipeline(
                "image-classification",
                model=wd14_model,
                image_processor=wd14_processor,
                device=device
            )

            # BLIP2 for captioning
            logger.info(f"Loading BLIP2 from {self.MODEL_CONFIGS['blip2']['cache_dir']}")
            self.classifiers['blip2_processor'] = BlipProcessor.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id'],
                cache_dir=self.MODEL_CONFIGS['blip2']['cache_dir']
            )
            
            # For BLIP2, we can try using MPS
            model_device = self.device
            dtype = torch.float16 if model_device in ['cuda', 'mps'] else torch.float32
            
            self.classifiers['blip2_model'] = BlipForConditionalGeneration.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id'],
                cache_dir=self.MODEL_CONFIGS['blip2']['cache_dir'],
                torch_dtype=dtype
            ).to(model_device)

            logger.info("All classifiers loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load classifiers: {e}")
            raise

    async def get_orientation(self, image_path: Union[str, Path]) -> List[TagResult]:
        """Get orientation tags for an image"""
        image_path = str(image_path) if isinstance(image_path, Path) else image_path
        try:
            results = self.classifiers['orientation'](image_path)
            return [
                TagResult(
                    label=result['label'],
                    confidence=result['score'],
                    category='orientation'
                ) for result in results
            ]
        except Exception as e:
            logger.error(f"Error getting orientation for {image_path}: {e}")
            return []

    async def get_wd14_tags(self, image_path: Union[str, Path]) -> List[TagResult]:
        """Get WD14 tags for an image"""
        image_path = str(image_path) if isinstance(image_path, Path) else image_path
        try:
            results = self.classifiers['wd14'](image_path)
            # Filter and sort by confidence
            filtered_results = [
                TagResult(
                    label=result['label'],
                    confidence=result['score'],
                    category='wd14'
                )
                for result in results
                if result['score'] > self.threshold
            ]
            return sorted(filtered_results, key=lambda x: x.confidence, reverse=True)
        except Exception as e:
            logger.error(f"Error getting WD14 tags for {image_path}: {e}")
            return []

    async def generate_caption(self, image_path: Union[str, Path]) -> str:
        """Generate a caption for the image using BLIP2"""
        try:
            # Load and preprocess the image
            image = Image.open(str(image_path)).convert('RGB')
            inputs = self.classifiers['blip2_processor'](image, return_tensors="pt").to(self.device)

            # Generate caption
            output = self.classifiers['blip2_model'].generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                temperature=1.0
            )
            caption = self.classifiers['blip2_processor'].decode(output[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return ""

    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run all available taggers on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all tags and caption
        """
        results = {}
        
        # Get orientation
        orientation_results = await self.get_orientation(image_path)
        if orientation_results:
            results['orientation'] = orientation_results

        # Get WD14 tags
        wd14_results = await self.get_wd14_tags(image_path)
        if wd14_results:
            results['wd14'] = wd14_results

        # Generate caption
        caption = await self.generate_caption(image_path)
        if caption:
            results['caption'] = caption
        
        return results

    def add_classifier(self, name: str, classifier: Any):
        """Add a new classifier to the tagger"""
        self.classifiers[name] = classifier
        logger.info(f"Added new classifier: {name}")
