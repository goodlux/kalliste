from pathlib import Path
from typing import List, Dict, Optional
import logging
import yaml
from copy import deepcopy
from .original_image import OriginalImage

logger = logging.getLogger(__name__)

class Batch:
    def __init__(self, input_path: Path, output_path: Path, default_config: Dict):
        self.input_path = input_path
        self.output_path = output_path
        self.default_config = default_config
        self.config = self._load_batch_config()
        self.images: List[OriginalImage] = []

    def _load_batch_config(self) -> Dict:
        config = deepcopy(self.default_config)
        batch_config_path = self.input_path / 'detection_config.yaml'
        if batch_config_path.exists():
            try:
                logger.info(f"Found batch-specific config at {batch_config_path}")
                with open(batch_config_path) as f:
                    batch_config = yaml.safe_load(f)
                config.update(batch_config)
                logger.info("Merged batch-specific config with defaults")
            except Exception as e:
                logger.error(f"Failed to load batch config from {batch_config_path}", exc_info=True)
                
        return config

    def scan_for_images(self):
        logger.info(f"Scanning for images in {self.input_path}")
        base_formats = ['.jpg', '.jpeg', '.png', '.dng']
        
        try:
            found_images = []
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
        logger.info(f"Processing batch: {self.input_path.name}")
        
        if not self.images:
            self.scan_for_images()
            
        if not self.images:
            logger.warning("No images to process")
            return

        # Process images sequentially
        for image in self.images:
            try:
                logger.info(f"Processing image: {image.source_path.name}")
                await image.process()
            except Exception as e:
                logger.error(f"Failed to process image {image.source_path}", exc_info=True)
                raise