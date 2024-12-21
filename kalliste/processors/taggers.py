"""
Image tagging and classification system for Kalliste.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from transformers import (
    pipeline, 
    Blip2Processor, 
    Blip2ForConditionalGeneration, 
    AutoModelForImageClassification, 
    AutoProcessor
)
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import logging
import numpy as np
from PIL import Image
import os
import timm
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

def get_default_device():
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        try:
            # Test MPS with a small operation
            test_tensor = torch.zeros(1).to('mps')
            _ = test_tensor + 1
            logger.info("MPS device validated")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS failed validation: {e}")
            logger.info("Falling back to CPU")
            return "cpu"
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
            'model_id': "hf_hub:SmilingWolf/wd-vit-large-tagger-v3",
            'cache_dir': WEIGHTS_DIR / "wd14",
            'tags_file': "weights/wd14/selected_tags.csv",
            'model_type': "vit"
        },
        'blip2': {
            'model_id': "Salesforce/blip2-opt-2.7b",
            'cache_dir': WEIGHTS_DIR / "blip2"
        }
    }

    # Tags we want to filter out
    BLACKLISTED_TAGS = {
        'questionable', 'explicit', 'nude', 'nudity', 'nipples', 'pussy',
        'penis', 'sex', 'cum', 'penetration', 'penetrated'
    }
    
    def __init__(self, device: Optional[str] = None, blacklisted_tags: Optional[set] = None):
        """Initialize the ImageTagger with specified device."""
        self.device = device or get_default_device()
        logger.info(f"Initializing ImageTagger on device: {self.device}")
        
        # For models that need CPU when using MPS
        self.model_device = 'cpu' if self.device == 'mps' else self.device
        
        # Allow custom blacklist to be passed in
        self.blacklisted_tags = blacklisted_tags or self.BLACKLISTED_TAGS
        
        # Create weights directory if it doesn't exist
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        for config in self.MODEL_CONFIGS.values():
            config['cache_dir'].mkdir(parents=True, exist_ok=True)
        
        self.classifiers: Dict[str, Any] = {}
        self.threshold = 0.35  # Confidence threshold for WD14 tags
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialize all classifiers"""
        logger.info("Loading classifiers...")
        try:
            # Use CPU for certain operations that might be problematic on MPS
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

            # WD14 Tagger - Using local timm model
            logger.info("Loading WD14 tagger")
            try:
                # Load model using timm
                self.classifiers['wd14_model'] = timm.create_model(
                    self.MODEL_CONFIGS['wd14']['model_id'],
                    pretrained=True
                ).to('cpu')  # Always load WD14 to CPU
                self.classifiers['wd14_model'].eval()
                
                # Load tags
                tags_path = Path(self.MODEL_CONFIGS['wd14']['tags_file'])
                self.wd14_tags = np.genfromtxt(tags_path, delimiter=',', dtype=str)
                
                # Set up image transforms for WD14
                self.classifiers['wd14_transform'] = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"Loaded WD14 model and {len(self.wd14_tags)} tags")
                
            except Exception as e:
                logger.error(f"Failed to load WD14 tagger: {e}")
                logger.exception(e)
                raise

            # BLIP2 for captioning
            logger.info(f"Loading BLIP2 from {self.MODEL_CONFIGS['blip2']['cache_dir']}")
            self.classifiers['blip2_processor'] = Blip2Processor.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id'],
                cache_dir=self.MODEL_CONFIGS['blip2']['cache_dir']
            )
            
            # For BLIP2, we can try using MPS
            model_device = self.device
            dtype = torch.float16 if model_device in ['cuda', 'mps'] else torch.float32
            
            self.classifiers['blip2_model'] = Blip2ForConditionalGeneration.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id'],
                cache_dir=self.MODEL_CONFIGS['blip2']['cache_dir'],
                torch_dtype=dtype,
                load_in_8bit=True if model_device == 'cuda' else False
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

    async def get_wd14_tags(self, image_path: Union[str, Path], num_tags: int = 10) -> List[TagResult]:
        """Get WD14 tags for an image"""
        try:
            # Load and preprocess image
            image = Image.open(str(image_path)).convert('RGB')
            
            # Transform image on CPU
            image_tensor = self.classifiers['wd14_transform'](image)
            image_tensor = image_tensor.unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                output = self.classifiers['wd14_model'](image_tensor)
                probs = torch.sigmoid(output)
            
            # Convert to numpy for processing
            probs = probs.cpu().numpy()[0]
            
            # Convert to tag results, filtering out blacklisted tags
            results = []
            for i, (conf, tag) in enumerate(zip(probs, self.wd14_tags)):
                if conf > self.threshold:
                    # Handle both string and array tag formats
                    tag_name = tag[1] if isinstance(tag, np.ndarray) else tag
                    # Clean up tag name
                    tag_name = str(tag_name).strip()
                    # Skip blacklisted tags
                    if tag_name and not tag_name.startswith('tag_id') and tag_name.lower() not in self.blacklisted_tags:
                        results.append(TagResult(
                            label=tag_name,
                            confidence=float(conf),
                            category='wd14'
                        ))
            
            # Return top N tags by confidence
            return sorted(results, key=lambda x: x.confidence, reverse=True)[:num_tags]
            
        except Exception as e:
            logger.error(f"Error getting WD14 tags for {image_path}: {e}")
            logger.exception(e)  # Log full traceback for debugging
            return []

    async def generate_caption(self, image_path: Union[str, Path]) -> str:
        """Generate a caption for the image using BLIP2"""
        try:
            # Load and preprocess the image
            image = Image.open(str(image_path)).convert('RGB')
            inputs = self.classifiers['blip2_processor'](image, return_tensors="pt").to(self.device)

            # Generate caption with improved parameters
            output = self.classifiers['blip2_model'].generate(
                **inputs,
                do_sample=True,  # Enable sampling for more natural captions
                max_new_tokens=50,
                min_length=10,
                num_beams=5,
                length_penalty=1.0,  # Favor slightly longer captions
                temperature=0.7,    
                top_k=50,          
                top_p=0.9,
                repetition_penalty=1.2  # Discourage repetition
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
