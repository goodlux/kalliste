from pathlib import Path
from typing import List, Dict, Optional
import logging
import yaml
import asyncio
from copy import deepcopy
from .original_image import OriginalImage

logger = logging.getLogger(__name__)

class Batch:
    def __init__(self, input_path: Path, output_path: Path, default_config: Dict, shutdown_event: Optional[asyncio.Event] = None):
        self.input_path = input_path
        self.output_path = output_path
        self.default_config = default_config
        self.config = self._load_batch_config()
        self.images: List[OriginalImage] = []
        self.shutdown_event = shutdown_event

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
        """Process all images in the batch concurrently with limits."""
        logger.info(f"Processing batch: {self.input_path.name}")
        
        if not self.images:
            self.scan_for_images()
            
        if not self.images:
            logger.warning("No images to process")
            return
            
        # Get concurrency limit from config, default to 4
        max_concurrent = self.config.get('processing', {}).get('max_concurrent_images', 4)
        
        # Process images concurrently with semaphore
        sem = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image):
            if self.shutdown_event and self.shutdown_event.is_set():
                logger.info("Shutdown requested, skipping image processing")
                return
                
            async with sem:
                try:
                    logger.info(f"Processing image: {image.source_path.name}")
                    await image.process()
                except Exception as e:
                    logger.error(f"Failed to process image {image.source_path}", exc_info=True)
                    raise
        
        # Create tasks for all images
        tasks = []
        for image in self.images:
            if self.shutdown_event and self.shutdown_event.is_set():
                break
            # Create a task from the coroutine
            task = asyncio.create_task(process_with_semaphore(image))
            tasks.append(task)
        
        if not tasks:
            return
            
        # Wait for tasks with cancellation support
        try:
            # Use gather to handle all tasks together
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Task failed in batch {self.input_path.name}: {e}")
            raise
        finally:
            # Cancel any remaining tasks if shutdown requested
            if self.shutdown_event and self.shutdown_event.is_set():
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait briefly for cancellation
                await asyncio.wait(tasks, timeout=2.0)