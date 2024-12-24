from pathlib import Path
from typing import List
from .original_image import OriginalImage
from .types import ProcessingStatus
import asyncio
import logging

logger = logging.getLogger(__name__)

class Batch:
    def __init__(self, input_path: Path, output_path: Path, max_concurrent: int = 3):
        logger.info(f"Initializing Batch with input_path={input_path}, max_concurrent={max_concurrent}")
        self.input_path = input_path
        self.output_path = output_path
        self.original_images: List[OriginalImage] = []
        self.status = ProcessingStatus.PENDING
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.scan_for_images()
        
    def scan_for_images(self):
        """Find all images in batch folder"""
        logger.info(f"Scanning for images in {self.input_path}")
        image_count = 0
        
        for extension in ['*.png', '*.jpg', '*.jpeg']:
            logger.debug(f"Scanning for {extension} files")
            for file in self.input_path.glob(extension):
                logger.debug(f"Found image: {file}")
                self.original_images.append(OriginalImage(
                    input_path=file,
                    output_path=self.output_path
                ))
                image_count += 1
                    
        logger.info(f"Found {image_count} images to process in batch {self.input_path.name}")
                
    async def process_image(self, image: OriginalImage):
        """Process a single image with semaphore control"""
        logger.debug(f"Waiting for semaphore to process {image.input_path.name}")
        async with self.semaphore:
            logger.debug(f"Acquired semaphore for {image.input_path.name}")
            try:
                return await image.process()
            except Exception as e:
                # Log but don't wrap - error context is already included from image.process()
                logger.error(f"Failed to process {image.input_path.name}", exc_info=True)
                raise
            finally:
                logger.debug(f"Released semaphore for {image.input_path.name}")
            
    async def process(self):
        """Process images in batch with concurrency limit"""
        logger.info(f"Starting batch processing for {self.input_path.name} with {len(self.original_images)} images")
        self.status = ProcessingStatus.PROCESSING
        
        try:
            if not self.original_images:
                logger.warning(f"No images found in batch {self.input_path}")
                return []
            
            # Create tasks but limit concurrent execution
            logger.debug("Creating processing tasks")
            tasks = [self.process_image(image) for image in self.original_images]
            
            # Use gather without wrapping errors
            logger.debug("Starting gather operation")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for any errors in results
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.error(f"Batch processing completed with {len(errors)} errors")
                # Raise the first error without wrapping
                raise errors[0]
            
            logger.info(f"Successfully processed {len(results)} images in batch {self.input_path.name}")
            self.status = ProcessingStatus.COMPLETE
            await self.on_complete()
            return results
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Batch processing failed", exc_info=True)
            raise  # Don't wrap, just propagate the original error
        
    async def on_complete(self):
        """Called when all images in batch are complete"""
        logger.info(f"Batch {self.input_path} complete")
        
    async def on_image_complete(self, image_path: Path):
        """Called when an original image signals completion"""
        logger.info(f"Image {image_path} in batch {self.input_path} complete")