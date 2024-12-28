"""Pipeline for managing and running multiple image taggers."""
from typing import Dict, List, Optional, Type, Union, Set
import asyncio
from pathlib import Path
import logging
from dataclasses import dataclass, field
from transformers import (
    AutoModelForImageClassification, 
    AutoProcessor,
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
import timm
from .base_tagger import BaseTagger
from ..types import TagResult
from .caption_tagger import CaptionTagger
from .wd14_tagger import WD14Tagger
from .orientation_tagger import OrientationTagger
from ..config import (
    ORIENTATION_MODEL_ID,
    BLIP2_MODEL_ID,
    WD14_MODEL_ID
)

logger = logging.getLogger(__name__)

@dataclass
class TaggerTypeConfig:
    """Configuration for a specific detector type (face, person, etc)"""
    enabled_taggers: Set[str]
    wd14_confidence: float = 0.35
    wd14_max_tags: int = 50
    wd14_min_tags: int = 5

@dataclass
class PipelineConfig:
    """Configuration for tagger pipeline."""
    default_config: TaggerTypeConfig = field(default_factory=lambda: TaggerTypeConfig(
        enabled_taggers={'caption', 'wd14', 'orientation'}
    ))
    type_configs: Dict[str, TaggerTypeConfig] = field(default_factory=dict)

    def get_config_for_type(self, detection_type: str) -> TaggerTypeConfig:
        """Get tagger configuration for a specific detection type"""
        return self.type_configs.get(detection_type, self.default_config)

class TaggerPipeline:
    """Pipeline for managing and running multiple image taggers."""
    
    def __init__(self, config: Dict):
        """Initialize tagger pipeline with configuration.
        
        Args:
            config: Configuration dictionary from detection_config.yaml
        """
        self.config = config
        self.taggers = {}
        
    async def _ensure_taggers_initialized(self, region_type: str):
        """Initialize taggers needed for this region type."""
        # Get which taggers we need for this region type
        needed_taggers = set(self.config[region_type])
        
        for tagger_name in needed_taggers - set(self.taggers.keys()):
            try:
                if tagger_name == 'wd14':
                    model = timm.create_model("hf_hub:SmilingWolf/wd-vit-large-tagger-v3", pretrained=True) #TODO: Standardize name to use model name in config, also loading with timm here to match model_registry.py.
                    #AutoModelForImageClassification.from_pretrained(WD14_MODEL_ID)
                    self.taggers['wd14'] = WD14Tagger(
                        model=model,
                        config=self.config
                    )
                    
                elif tagger_name == 'caption':
                    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_ID)
                    self.taggers['caption'] = CaptionTagger(
                        model=model,
                        config=self.config
                    )
                    
                elif tagger_name == 'orientation':
                    model = AutoModelForImageClassification.from_pretrained(ORIENTATION_MODEL_ID)
                    self.taggers['orientation'] = OrientationTagger(
                        model=model,
                        config=self.config
                    )
                    
            except Exception as e:
                logger.error(f"Failed to initialize {tagger_name} tagger: {e}", exc_info=True)
                raise RuntimeError(f"Tagger initialization failed: {e}")
    
    async def tag_image(self, image_path: Path, region_type: str) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type.
        
        Args:
            image_path: Path to image to tag
            region_type: Type of region (e.g., 'person', 'face')
            
        Returns:
            Dictionary mapping tagger names to their results
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If tagging fails
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Ensure we have the taggers we need
        await self._ensure_taggers_initialized(region_type)
        
        # Get list of taggers to run for this region type
        taggers_to_run = self.config[region_type]
        
        results = {}
        try:
            # Run each configured tagger
            for tagger_name in taggers_to_run:
                if tagger_name in self.taggers:
                    tagger = self.taggers[tagger_name]
                    results.update(await tagger.tag_image(image_path))
                else:
                    logger.warning(f"Tagger {tagger_name} not available")
                    
            return results
            
        except Exception as e:
            logger.error(f"Tagging failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Tagging failed: {str(e)}") from e