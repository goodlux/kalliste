"""Handles batch-level image processing."""
from pathlib import Path
from typing import List, Dict
import logging
import yaml
from copy import deepcopy

from ..image.original_image import OriginalImage
from .batch_statistics import BatchStatistics
from .batch_results_writer import BatchResultsWriter

logger = logging.getLogger(__name__)

class Batch:
    def __init__(self, input_path: Path, output_path: Path, default_config: Dict):
        self.input_path = input_path
        self.output_path = output_path
        self.default_config = default_config
        self.config = self._load_batch_config()
        self.images: List[OriginalImage] = []
        self.stats = BatchStatistics()

    def _load_batch_config(self) -> Dict:
        """Load batch-specific config, falling back to defaults."""
        # Start with a deep copy of default config
        config = deepcopy(self.default_config)
        
        # Check for batch-specific config
        batch_config_path = self.input_path / 'detection_config.yaml'
        if batch_config_path.exists():
            try:
                logger.info(f"Found batch-specific config at {batch_config_path}")
                with open(batch_config_path) as f:
                    batch_config = yaml.safe_load(f)
                    
                # Merge batch config over defaults
                config.update(batch_config)
                logger.info("Merged batch-specific config with defaults")
            except Exception as e:
                logger.error(f"Failed to load batch config from {batch_config_path}", exc_info=True)
                # Continue with defaults
                
        return config

    def scan_for_images(self):
        """Scan input directory for supported image formats."""
        logger.info(f"Scanning for images in {self.input_path}")
        
        # Base formats - we'll handle case variations
        base_formats = ['.jpg', '.jpeg', '.png', '.dng']
        
        try:
            found_images = []
            # Use rglob to get all files and filter by extension
            for file in self.input_path.rglob('*'):
                if file.suffix.lower() in base_formats:
                    try:
                        original_image = OriginalImage(
                            source_path=file,
                            output_dir=self.output_path,
                            config=self.config
                        )
                        found_images.append(original_image)
                        logger.debug(f"Added image: {file}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create OriginalImage for {file}", exc_info=True)
                        continue
            
            self.images = found_images
                    
            if not self.images:
                logger.warning(f"No supported images found in {self.input_path}")
                logger.info(f"Supported formats (case insensitive): {', '.join(base_formats)}")
                
            logger.info(f"Found {len(self.images)} images")
                    
        except Exception as e:
            logger.error(f"Error scanning for images", exc_info=True)
            raise

    async def process(self):
        """Process all images in the batch."""
        logger.info(f"Processing batch: {self.input_path.name}")
        
        # Scan for images if not already done
        if not self.images:
            self.scan_for_images()
            
        if not self.images:
            logger.warning("No images to process")
            return
            
        # Process each image
        for image in self.images:
            try:
                logger.info(f"Processing image: {image.source_path.name}")
                self.stats.increment_original_images()
                
                # Process image and get results
                image_results = await image.process()
                if image_results:
                    self._update_statistics(image_results)
                    
            except Exception as e:
                logger.error(f"Failed to process image {image.source_path}", exc_info=True)
                raise
                
        # Write batch results
        try:
            writer = BatchResultsWriter(self.output_path)
            writer.write_results(self.stats)
        except Exception as e:
            logger.error("Failed to write batch results", exc_info=True)
            
        logger.info(f"Completed processing batch: {self.input_path.name}")

    def _update_statistics(self, image_results: Dict):
        """Update batch statistics with results from an image."""
        if not image_results:
            return
            
        # Update region statistics
        for region_type, region_stats in image_results.get('regions', {}).items():
            # Record detected region
            self.stats.add_region(region_type)
            
            # Record if rejected for size
            if region_stats.get('rejected_small', False):
                self.stats.add_small_region(region_type)
                
            # Record assessments if present
            if all(key in region_stats for key in ['technical', 'aesthetic', 'overall', 'kalliste']):
                self.stats.add_assessments(
                    region_type=region_type,
                    technical=region_stats['technical'],
                    aesthetic=region_stats['aesthetic'],
                    overall=region_stats['overall'],
                    kalliste=region_stats['kalliste']
                )