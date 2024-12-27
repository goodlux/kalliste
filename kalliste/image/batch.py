from pathlib import Path
from typing import List, Dict
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
        """Scan input directory for images."""
        logger.info(f"Scanning for images in {self.input_path}")
        
        try:
            for file in self.input_path.glob('*.png'):
                try:
                    original_image = OriginalImage(
                        source_path=file,
                        output_dir=self.output_path,
                        config=self.config
                    )
                    self.images.append(original_image)
                    
                except Exception as e:
                    logger.error(f"Failed to create OriginalImage for {file}", exc_info=True)
                    continue
                    
            if not self.images:
                logger.warning(f"No PNG images found in {self.input_path}")
                
            logger.info(f"Found {len(self.images)} images")
                    
        except Exception as e:
            logger.error(f"Error scanning for images", exc_info=True)
            raise