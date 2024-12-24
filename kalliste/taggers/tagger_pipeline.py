"""Pipeline for managing and running multiple image taggers."""

from typing import Dict, List, Optional, Type, Union, Set
import asyncio
from pathlib import Path
import logging
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModelForImageClassification, 
    AutoProcessor,
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
import timm
import pandas as pd

from .base_tagger import BaseTagger, get_default_device
from ..image.types import TagResult
from .caption_tagger import CaptionTagger
from .wd14_tagger import WD14Tagger
from .orientation_tagger import OrientationTagger
from ..config import (
    ORIENTATION_MODEL_ID,
    BLIP2_MODEL_ID,
    WD14_MODEL_ID,
    WD14_TAGS_FILE
)

logger = logging.getLogger(__name__)

@dataclass
class TaggerTypeConfig:
    """Configuration for a specific detector type (face, person, etc)"""
    enabled_taggers: Set[str]
    wd14_confidence: float = 0.35
    wd14_categories: Optional[List[str]] = None
    wd14_blacklist: Optional[set] = None

@dataclass
class PipelineConfig:
    """Configuration for tagger pipeline."""
    device: Optional[str] = None
    default_config: TaggerTypeConfig = field(default_factory=lambda: TaggerTypeConfig(
        enabled_taggers={'caption', 'wd14', 'orientation'}
    ))
    type_configs: Dict[str, TaggerTypeConfig] = field(default_factory=dict)

    def get_config_for_type(self, detection_type: str) -> TaggerTypeConfig:
        """Get tagger configuration for a specific detection type"""
        return self.type_configs.get(detection_type, self.default_config)

class TaggerPipeline:
    """Manages multiple image taggers and combines their results."""
    
    TAGGER_CLASSES = {
        'caption': CaptionTagger,
        'wd14': WD14Tagger,
        'orientation': OrientationTagger
    }

    # Default configurations for different detection types
    DEFAULT_TYPE_CONFIGS = {
        'face': TaggerTypeConfig(
            enabled_taggers={'orientation'},  # Faces only need orientation
            wd14_confidence=0.5
        ),
        'person': TaggerTypeConfig(
            enabled_taggers={'caption', 'wd14', 'orientation'},
            wd14_confidence=0.35,
            wd14_categories=['clothing', 'pose', 'action']
        )
    }
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the tagger pipeline."""
        logger.info("Creating TaggerPipeline instance")
        self.config = config or PipelineConfig(
            type_configs=self.DEFAULT_TYPE_CONFIGS.copy()
        )
        self.device = self.config.device or get_default_device()
        self.taggers: Dict[str, BaseTagger] = {}
        logger.info(f"TaggerPipeline created with device: {self.device}")
    
    async def initialize(self):
        """Initialize all required models and create tagger instances."""
        logger.info("Starting tagger pipeline initialization")
        
        # Initialize orientation tagger
        logger.info("Initializing orientation tagger...")
        try:
            device = 'cpu' if self.device == 'mps' else self.device
            orientation_model = AutoModelForImageClassification.from_pretrained(
                ORIENTATION_MODEL_ID
            ).to(device)
            orientation_processor = AutoProcessor.from_pretrained(ORIENTATION_MODEL_ID)
            
            self.taggers['orientation'] = OrientationTagger(
                model=orientation_model,
                processor=orientation_processor,
                device=self.device
            )
            logger.info("Orientation tagger created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orientation tagger: {e}")
            raise
            
        # Initialize caption tagger
        logger.info("Initializing caption tagger...")
        try:
            device = "cpu" if self.device == "mps" else self.device
            dtype = torch.float16 if device == 'cuda' else torch.float32
            
            caption_processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
            caption_model = Blip2ForConditionalGeneration.from_pretrained(
                BLIP2_MODEL_ID,
                torch_dtype=dtype,
                load_in_8bit=device == 'cuda'
            ).to(device)
            
            self.taggers['caption'] = CaptionTagger(
                model=caption_model,
                processor=caption_processor,
                device=self.device
            )
            logger.info("Caption tagger created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize caption tagger: {e}")
            raise
            
        # Initialize WD14 tagger
        logger.info("Initializing WD14 tagger...")
        try:
            wd14_model = timm.create_model(
                WD14_MODEL_ID,
                pretrained=True
            ).to('cpu')  # WD14 always uses CPU
            wd14_model.eval()
            
            # Load tag definitions
            tags_file = Path(WD14_TAGS_FILE)
            if not tags_file.exists():
                raise FileNotFoundError(f"Tags file not found: {tags_file}")
            
            tags_df = pd.read_csv(tags_file)
            logger.info(f"Loaded {len(tags_df)} WD14 tags")
            
            self.taggers['wd14'] = WD14Tagger(
                model=wd14_model,
                tags_df=tags_df,
                device=self.device,
                confidence_threshold=self.config.default_config.wd14_confidence
            )
            logger.info("WD14 tagger created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WD14 tagger: {e}")
            raise
            
        logger.info("All taggers initialized successfully")
    
    async def tag_image(self, 
                       image_path: Union[str, Path],
                       detection_type: str) -> Dict[str, List[TagResult]]:
        """Run appropriate taggers for the detection type on an image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Get configuration for this detection type
        type_config = self.config.get_config_for_type(detection_type)
        
        # Create tasks for enabled taggers
        tasks = []
        for name in type_config.enabled_taggers:
            if name not in self.taggers:
                logger.warning(f"Tagger {name} not initialized, skipping")
                continue
                
            tagger = self.taggers[name]
            # Apply type-specific configuration for WD14
            if name == 'wd14':
                tagger.confidence_threshold = type_config.wd14_confidence
                tagger.category_filters = type_config.wd14_categories
                tagger.blacklist = type_config.wd14_blacklist
            tasks.append(self._run_tagger(name, tagger, image_path))
        
        # Run enabled taggers in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and process results
        combined_results: Dict[str, List[TagResult]] = {}
        for name, result in zip(type_config.enabled_taggers, results):
            if isinstance(result, Exception):
                logger.error(f"Tagger {name} failed: {result}")
                combined_results.update(self._get_empty_result(name))
            else:
                combined_results.update(result)
        
        return combined_results
    
    async def _run_tagger(
        self,
        name: str,
        tagger: BaseTagger,
        image_path: Path
    ) -> Dict[str, List[TagResult]]:
        """Run a single tagger with error handling."""
        try:
            logger.info(f"Running {name} tagger on {image_path}")
            return await tagger.tag_image(image_path)
        except Exception as e:
            logger.error(f"Error in {name} tagger: {e}")
            raise
    
    def _get_empty_result(self, tagger_name: str) -> Dict[str, List[TagResult]]:
        """Get empty results for a failed tagger."""
        if tagger_name == 'caption':
            return {
                'caption': [
                    TagResult(
                        label="Failed to generate caption",
                        confidence=0.0,
                        category='caption'
                    )
                ]
            }
        elif tagger_name == 'wd14':
            return {'wd14': []}
        elif tagger_name == 'orientation':
            return {
                'orientation': [
                    TagResult(
                        label="unknown",
                        confidence=0.0,
                        category='orientation'
                    )
                ]
            }
        return {}