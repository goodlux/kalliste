"""Pipeline for managing and running multiple image taggers."""
from typing import Dict, List, Optional
from pathlib import Path
import logging
from PIL import Image

from .AbstractTagger import AbstractTagger
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
        self.taggers: Dict[str, AbstractTagger] = {}
        
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
    
    async def tag_pillow_image(self, image: Image.Image, region_type: str) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type using a PIL Image."""
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
                    logger.debug(f"Running {tagger_name} tagger")
                    tagger_results = await self.taggers[tagger_name].tag_pillow_image(image)
                    
                    # Log individual tagger results
                    for category, tags in tagger_results.items():
                        if tags:  # Only log if we have tags
                            tag_details = [
                                f"{tag.label}({tag.confidence:.2f})" 
                                for tag in tags
                            ]
                            logger.debug(f"{tagger_name} {category} results: {tag_details}")
                    
                    results.update(tagger_results)
                else:
                    logger.warning(f"Unsupported tagger requested: {tagger_name}")
            
            # Log final combined results
            combined_results = []
            for category, tags in results.items():
                if tags:  # Only include non-empty results
                    if category == 'caption':
                        # Special formatting for captions
                        formatted = f"{category}: \"{tags[0].label}\""
                    else:
                        # Format other tags with confidence
                        formatted = f"{category}: " + ", ".join(
                            f"{tag.label}({tag.confidence:.2f})"
                            for tag in tags
                        )
                    combined_results.append(formatted)
                    
            if combined_results:
                logger.info("Tagger results:")
                for result in combined_results:
                    logger.info(f"  {result}")
            else:
                logger.warning("No tag results generated")
                    
            return results
            
        except Exception as e:
            logger.error(f"Tagging failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Tagging failed: {str(e)}") from e
            
    async def tag_image(self, image_path: Path, region_type: str) -> Dict[str, List[TagResult]]:
        """Run configured taggers for this region type using an image file.
        
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
            
        with Image.open(image_path) as img:
            return await self.tag_pillow_image(img, region_type)