"""Pipeline for managing and running multiple image taggers."""
from typing import Dict, List, Optional
from pathlib import Path
import logging

from .base_tagger import BaseTagger
from .wd14_tagger import WD14Tagger
from .caption_tagger import CaptionTagger
from .orientation_tagger import OrientationTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class TaggerPipeline:
    """Pipeline for managing and running multiple image taggers.
    
    This pipeline coordinates running the appropriate taggers based on the
    detection type and configuration. Each tagger gets its model from the
    model registry and uses configuration specified in the detection config.
    """
    
    # Map of tagger name to tagger class
    TAGGER_CLASSES = {
        'wd14': WD14Tagger,
        'caption': CaptionTagger,
        'orientation': OrientationTagger
    }
    
    def __init__(self, config: Dict):
        """Initialize tagger pipeline with configuration.
        
        Args:
            config: Configuration dictionary from detection_config.yaml
                Should contain tagger-specific configs and type mappings
        """
        self.config = config
        self.taggers: Dict[str, BaseTagger] = {}
        
    def _ensure_tagger_initialized(self, tagger_name: str) -> None:
        """Initialize a specific tagger if not already initialized.
        
        Args:
            tagger_name: Name of tagger to initialize
            
        Raises:
            ValueError: If tagger_name is not supported
            RuntimeError: If tagger initialization fails
        """
        if tagger_name not in self.taggers:
            if tagger_name not in self.TAGGER_CLASSES:
                raise ValueError(f"Unsupported tagger: {tagger_name}")
                
            try:
                # Get tagger-specific config if it exists
                tagger_config = self.config.get('tagger', {}).get(tagger_name, {})
                
                # Initialize tagger with config
                tagger_class = self.TAGGER_CLASSES[tagger_name]
                self.taggers[tagger_name] = tagger_class(config=tagger_config)
                
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
            ValueError: If region_type has no configured taggers
            RuntimeError: If tagging fails
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Get list of taggers to run for this region type
        taggers_to_run = self.config.get(region_type)
        if not taggers_to_run:
            raise ValueError(f"No taggers configured for region type: {region_type}")
        
        results = {}
        try:
            # Initialize and run each configured tagger
            for tagger_name in taggers_to_run:
                if tagger_name in self.TAGGER_CLASSES:
                    # Ensure tagger is initialized
                    self._ensure_tagger_initialized(tagger_name)
                    
                    # Run tagger
                    tagger_results = await self.taggers[tagger_name].tag_image(image_path)
                    results.update(tagger_results)
                else:
                    logger.warning(f"Unsupported tagger requested: {tagger_name}")
                    
            return results
            
        except Exception as e:
            logger.error(f"Tagging failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Tagging failed: {str(e)}") from e