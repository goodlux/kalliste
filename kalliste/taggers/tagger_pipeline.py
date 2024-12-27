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
from ..types import TagResult
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
    """Pipeline for managing and running multiple image taggers."""
    
    def __init__(self, config: Dict):
        """Initialize tagger pipeline with configuration."""
        self.config = config
        self.taggers = {}
        
    async def _ensure_taggers_initialized(self, region_type: str):
        """Initialize taggers needed for this region type."""
        # Get which taggers we need for this region type
        needed_taggers = set(self.config[region_type])
        
        for tagger_name in needed_taggers - set(self.taggers.keys()):
            if tagger_name == 'wd14':
                self.taggers['wd14'] = WD14Tagger(
                    config=self.config  # Pass full config
                )
            elif tagger_name == 'caption':
                self.taggers['caption'] = CaptionTagger(
                    config=self.config  # Pass full config
                )
            elif tagger_name == 'orientation':
                self.taggers['orientation'] = OrientationTagger(
                    config=self.config  # Pass full config
                )
        
    async def tag_image(self, image_path: Path, region_type: str) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type.
        
        Args:
            image_path: Path to image to tag
            region_type: Type of region (e.g., 'person', 'face')
            
        Returns:
            Dictionary mapping tagger names to their results
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