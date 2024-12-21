"""Image tagging and classification system for Kalliste."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from transformers import (
    pipeline, 
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
from pathlib import Path
import torch
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

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
    """Manages multiple image classification and tagging models."""
    
    MODEL_CONFIGS = {
        'orientation': {
            'model_id': "LucyintheSky/pose-estimation-front-side-back",
        },
        'wd14': {
            'model_id': "SmilingWolf/wd-v1-4-vit-tagger-v2",
            'model_file': "model.onnx",
            'tags_file': "selected_tags.csv"
        },
        'blip': {
            'model_id': "Salesforce/blip2-opt-2.7b",
        }
    }
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the ImageTagger with specified device."""
        self.device = device or get_default_device()
        logger.info(f"Initializing ImageTagger on device: {self.device}")
        
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
            logger.info("Loading orientation classifier")
            try:
                self.classifiers['orientation'] = pipeline(
                    "image-classification",
                    model=self.MODEL_CONFIGS['orientation']['model_id'],
                    device=device
                )
            except Exception as e:
                logger.error(f"Failed to load orientation classifier: {e}")
                raise

            # WD14 Tagger (ONNX)
            logger.info("Loading WD14 tagger")
            try:
                # Download ONNX model and tags
                model_path = hf_hub_download(
                    repo_id=self.MODEL_CONFIGS['wd14']['model_id'],
                    filename=self.MODEL_CONFIGS['wd14']['model_file']
                )
                tags_path = hf_hub_download(
                    repo_id=self.MODEL_CONFIGS['wd14']['model_id'],
                    filename=self.MODEL_CONFIGS['wd14']['tags_file']
                )
                
                # Load tags
                self.wd14_tags = np.genfromtxt(tags_path, delimiter=',', dtype=str)
                
                # Create ONNX Runtime session with appropriate provider
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                self.classifiers['wd14'] = ort.InferenceSession(model_path, providers=providers)
                
                # Get input shape info for preprocessing
                model_inputs = self.classifiers['wd14'].get_inputs()
                self.wd14_input_shape = model_inputs[0].shape
                logger.info(f"WD14 input shape: {self.wd14_input_shape}")
                
            except Exception as e:
                logger.error(f"Failed to load WD14 tagger: {e}")
                raise

            # BLIP2 for captioning
            logger.info("Loading BLIP2")
            try:
                self.classifiers['blip_processor'] = Blip2Processor.from_pretrained(
                    self.MODEL_CONFIGS['blip']['model_id']
                )
                
                # For BLIP2, we can try using MPS
                model_device = self.device
                dtype = torch.float16 if model_device in ['cuda', 'mps'] else torch.float32
                
                self.classifiers['blip_model'] = Blip2ForConditionalGeneration.from_pretrained(
                    self.MODEL_CONFIGS['blip']['model_id'],
                    torch_dtype=dtype,
                    load_in_8bit=True if model_device == 'cuda' else False
                ).to(model_device)
            except Exception as e:
                logger.error(f"Failed to load BLIP2: {e}")
                raise

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
        """Get WD14 tags for an image using ONNX model"""
        try:
            # Load and preprocess image
            image = Image.open(str(image_path)).convert('RGB')
            
            # Fixed size of 448x448 as per model requirements
            image = image.resize((448, 448), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch dimension if needed - maintaining NHWC format
            # [Height, Width, Channels] -> [1, Height, Width, Channels]
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, 0)
                
            # Run inference with properly formatted input
            outputs = self.classifiers['wd14'].run(
                None, 
                {'input_1:0': image_array}
            )
            probs = outputs[0][0]
            
            # Convert to tag results
            results = []
            for i, (conf, tag) in enumerate(zip(probs, self.wd14_tags)):
                if conf > self.threshold:
                    # Handle both string and array tag formats
                    tag_name = tag[1] if isinstance(tag, np.ndarray) else tag
                    # Clean up tag name
                    tag_name = str(tag_name).strip()
                    if tag_name and not tag_name.startswith('tag_id'):  # Skip metadata
                        results.append(TagResult(
                            label=tag_name,
                            confidence=float(conf),
                            category='wd14'
                        ))
            
            return sorted(results, key=lambda x: x.confidence, reverse=True)
        except Exception as e:
            logger.error(f"Error getting WD14 tags for {image_path}: {e}")
            return []

    async def generate_caption(self, image_path: Union[str, Path]) -> str:
        """Generate a caption for the image using BLIP2"""
        try:
            # Load and preprocess the image
            image = Image.open(str(image_path)).convert('RGB')
            inputs = self.classifiers['blip_processor'](image, return_tensors="pt").to(self.device)

            # Generate caption with BLIP2's improved parameters
            output = self.classifiers['blip_model'].generate(
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
            caption = self.classifiers['blip_processor'].decode(output[0], skip_special_tokens=True)
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