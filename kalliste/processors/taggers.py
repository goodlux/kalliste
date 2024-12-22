"""Image tagging and classification system for Kalliste."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from transformers import (
    pipeline, 
    Blip2Processor, 
    Blip2ForConditionalGeneration, 
    AutoModelForImageClassification, 
    AutoProcessor
)
from pathlib import Path
import torch
import logging
import pandas as pd
from PIL import Image
import timm
import torchvision.transforms as T

from ..config import (
    BLIP2_MODEL_ID,
    ORIENTATION_MODEL_ID,
    WD14_MODEL_ID,
    WD14_WEIGHTS_DIR,
    WD14_TAGS_FILE
)

logger = logging.getLogger(__name__)

def get_default_device():
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        try:
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
    """Manages multiple image classification and tagging models."""
    
    MODEL_CONFIGS = {
        'orientation': {
            'model_id': ORIENTATION_MODEL_ID,
            'model_type': "vit"
        },
        'wd14': {
            'model_id': WD14_MODEL_ID,
            'tags_file': WD14_TAGS_FILE,
            'model_type': "vit"
        },
        'blip2': {
            'model_id': BLIP2_MODEL_ID
        }
    }

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
        
        # Create WD14 weights directory if it doesn't exist
        WD14_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.classifiers: Dict[str, Any] = {}
        self.threshold = 0.35  # Confidence threshold for WD14 tags
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialize all classifiers"""
        logger.info("Loading classifiers...")
        try:
            # Use CPU for certain operations that might be problematic on MPS
            device = self.device if self.device != "mps" else "cpu"
            
            # Orientation classifier - uses default cache
            logger.info("Loading orientation classifier...")
            orientation_model = AutoModelForImageClassification.from_pretrained(
                self.MODEL_CONFIGS['orientation']['model_id']
            ).to(device)
            orientation_processor = AutoProcessor.from_pretrained(
                self.MODEL_CONFIGS['orientation']['model_id']
            )
            self.classifiers['orientation'] = pipeline(
                "image-classification",
                model=orientation_model,
                image_processor=orientation_processor,
                device=device
            )

            # WD14 Tagger - Loading directly from hub
            logger.info("Loading WD14 tagger...")
            try:
                # Load model using timm with hub reference
                self.classifiers['wd14_model'] = timm.create_model(
                    self.MODEL_CONFIGS['wd14']['model_id'],
                    pretrained=True
                ).to('cpu')  # Always load WD14 to CPU
                self.classifiers['wd14_model'].eval()
                
                # Load tags and extract just the 'name' column
                tags_path = Path(self.MODEL_CONFIGS['wd14']['tags_file'])
                if not tags_path.exists():
                    raise FileNotFoundError(f"WD14 tags file not found at {tags_path}")
                
                # Read CSV with pandas to properly handle the structure
                df = pd.read_csv(tags_path)
                # Convert tag names to a list of strings
                self.wd14_tags = df['name'].astype(str).tolist()
                
                logger.info(f"Loaded WD14 model and {len(self.wd14_tags)} tags")
                
                # Set up image transforms for WD14
                self.classifiers['wd14_transform'] = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                ])
                
            except Exception as e:
                logger.error(f"Failed to load WD14 tagger: {e}")
                logger.exception(e)
                raise

            # BLIP2 for captioning - uses default cache
            logger.info("Loading BLIP2...")
            self.classifiers['blip2_processor'] = Blip2Processor.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id']
            )
            
            # Use CPU for BLIP2 when MPS is active
            model_device = "cpu" if self.device == "mps" else self.device
            dtype = torch.float16 if model_device in ['cuda'] else torch.float32
            
            self.classifiers['blip2_model'] = Blip2ForConditionalGeneration.from_pretrained(
                self.MODEL_CONFIGS['blip2']['model_id'],
                torch_dtype=dtype,
                load_in_8bit=True if model_device == 'cuda' else False
            ).to(model_device)

            logger.info("All classifiers loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load classifiers: {e}")
            raise
            
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Tag an image with all available classifiers."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image once
        image = Image.open(image_path).convert('RGB')
        
        # Initialize results dictionary
        results: Dict[str, List[TagResult]] = {}
        
        # Run orientation classification
        if 'orientation' in self.classifiers:
            orientation_results = self.classifiers['orientation'](image)
            results['orientation'] = [
                TagResult(
                    label=result['label'],
                    confidence=result['score'],
                    category='orientation'
                )
                for result in orientation_results
            ]
        
        # Run WD14 tagging
        if 'wd14_model' in self.classifiers:
            # Prepare image
            img_tensor = self.classifiers['wd14_transform'](image)
            img_tensor = img_tensor.unsqueeze(0).to('cpu')  # Always process on CPU
            
            # Get predictions
            with torch.no_grad():
                output = self.classifiers['wd14_model'](img_tensor)
                probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            
            # Convert to TagResults, filter by threshold
            wd_results = []
            for i, (prob, tag) in enumerate(zip(probs, self.wd14_tags)):
                if prob >= self.threshold and tag not in self.blacklisted_tags:
                    wd_results.append(
                        TagResult(label=tag, confidence=float(prob), category='wd14')
                    )
            
            # Sort by confidence
            results['wd14'] = sorted(
                wd_results, 
                key=lambda x: x.confidence, 
                reverse=True
            )
        
        # Run BLIP2 captioning
        if 'blip2_model' in self.classifiers and 'blip2_processor' in self.classifiers:
            try:
                device = "cpu" if self.device == "mps" else self.device
                model = self.classifiers['blip2_model']
                processor = self.classifiers['blip2_processor']
                
                # Move model to CPU temporarily if using MPS
                if self.device == "mps":
                    model = model.to("cpu")
                
                # Process image
                inputs = processor(image, return_tensors="pt").to(device)
                
                # Generate caption
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=1,
                    length_penalty=1,
                    repetition_penalty=1.5
                )
                
                # Decode caption
                caption = processor.decode(output[0], skip_special_tokens=True)
                results['caption'] = caption.strip()
                
                # Move model back to MPS if needed
                if self.device == "mps":
                    model = model.to("mps")
                
            except Exception as e:
                logger.error(f"BLIP2 captioning failed: {e}")
                results['caption'] = "Failed to generate caption"
        
        return results