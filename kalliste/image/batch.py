from pathlib import Path
from typing import List
from .original_image import OriginalImage
from .types import ProcessingStatus
import asyncio
import logging
import traceback

logger = logging.getLogger(__name__)

class Batch:
    def __init__(self, input_path: Path, output_path: Path, max_concurrent: int = 3):
        logger.info(f"Initializing Batch with input_path={input_path}, max_concurrent={max_concurrent}")
        
        # Validate paths
        if not isinstance(input_path, Path):
            raise TypeError(f"input_path must be a Path object, got {type(input_path)}")
        if not isinstance(output_path, Path):
            raise TypeError(f"output_path must be a Path object, got {type(output_path)}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_path}")
            
        self.input_path = input_path
        self.output_path = output_path
        self.original_images: List[OriginalImage] = []
        self.status = ProcessingStatus.PENDING
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        try:
            self.scan_for_images()
        except Exception as e:
            logger.error(f"Failed to scan for images in {input_path}", exc_info=True)
            raise RuntimeError(f"Failed to initialize batch: {str(e)}") from e
        
    def scan_for_images(self):
        """Find all images in batch folder"""
        logger.info(f"Scanning for images in {self.input_path}")
        image_count = 0
        
        try:
            # Get list of supported extensions
            extensions = ['*.png', '*.jpg', '*.jpeg']
            
            for extension in extensions:
                logger.debug(f"Scanning for {extension} files")
                matching_files = list(self.input_path.glob(extension))
                
                for file in matching_files:
                    if not file.is_file():
                        logger.warning(f"Skipping non-file: {file}")
                        continue
                        
                    try:
                        logger.debug(f"Found image: {file}")
                        original_image = OriginalImage(
                            input_path=file,
                            output_path=self.output_path
                        )
                        self.original_images.append(original_image)
                        image_count += 1
                    except Exception as e:
                        logger.error(f"Failed to create OriginalImage for {file}", exc_info=True)
                        raise RuntimeError(f"Failed to initialize image {file}: {str(e)}") from e
                    
            if image_count == 0:
                logger.warning(f"No valid images found in batch {self.input_path.name}")
            else:
                logger.info(f"Found {image_count} images to process in batch {self.input_path.name}")
                
        except Exception as e:
            logger.error(f"Error scanning for images in {self.input_path}", exc_info=True)
            raise RuntimeError(f"Failed to scan for images: {str(e)}") from e
                
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