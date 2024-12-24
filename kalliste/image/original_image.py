from pathlib import Path
from typing import List, Optional
from .types import Detection, ProcessingStatus
from .cropped_image import CroppedImage
from ..detectors.base import DetectionConfig
from .model_registry import ModelRegistry
import asyncio
import shutil
import logging

logger = logging.getLogger(__name__)

class OriginalImage:
    # Default detection configurations
    DEFAULT_CONFIGS = [
        DetectionConfig(
            name="face",
            confidence_threshold=0.5,
            preferred_aspect_ratios=[(1, 1)]  # Square aspect ratio for faces
        ),
        DetectionConfig(
            name="person",
            confidence_threshold=0.5,
            preferred_aspect_ratios=[(2, 3), (3, 4)]  # Portrait ratios for people
        )
    ]
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.metadata = {}  # Original metadata
        self.cropped_images: List[CroppedImage] = []
        self.status = ProcessingStatus.PENDING
        
    async def process(self):
        """Main processing pipeline"""
        self.status = ProcessingStatus.PROCESSING
        logger.info(f"Starting processing for {self.input_path}")
        
        try:
            # First, copy original to output directory
            await self._copy_to_output()
            
            # Run detections
            try:
                detections = await self.run_detectors()
                logger.info(f"Found {len(detections)} detections in {self.input_path}")
            except Exception as e:
                logger.error(f"Detection failed for {self.input_path}: {e}", exc_info=True)
                raise  # Don't wrap, just propagate
            
            if detections:
                # Create and process crops asynchronously
                crop_tasks = []
                for detection in detections:
                    cropped = CroppedImage(
                        source_path=self.input_path,
                        output_dir=self.output_path,
                        detection=detection
                    )
                    self.cropped_images.append(cropped)
                    crop_tasks.append(cropped.process())
                    
                # Wait for all crops to complete
                try:
                    await asyncio.gather(*crop_tasks)
                except Exception as e:
                    logger.error(f"Error processing crops for {self.input_path}: {e}", exc_info=True)
                    raise  # Don't wrap, just propagate
                
            self.status = ProcessingStatus.COMPLETE
            logger.info(f"Completed processing for {self.input_path}")
            # Signal completion
            await self.on_complete()
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Error processing {self.input_path}: {e}", exc_info=True)
            raise  # Don't wrap, just propagate
            
    async def _copy_to_output(self):
        """Copy original image to output directory"""
        output_file = self.output_path / self.input_path.name
        try:
            # Run copy in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, shutil.copy2, str(self.input_path), str(output_file))
            logger.debug(f"Copied {self.input_path} to {output_file}")
        except Exception as e:
            logger.error(f"Failed to copy {self.input_path} to output: {e}", exc_info=True)
            # Provide specific error about what failed
            raise RuntimeError(f"Failed to copy image to {output_file}: {e}")
        
    async def run_detectors(self) -> List[Detection]:
        """Run all detectors on image asynchronously"""
        try:
            # Get detector from registry
            detector = ModelRegistry.get_detector()
            logger.debug(f"Got detector from registry for {self.input_path}")
            
            # Run detector in an executor to not block
            loop = asyncio.get_event_loop()
            regions = await loop.run_in_executor(
                None, 
                detector.detect,
                self.input_path
            )
            
            # Convert regions to our Detection type
            detections = [Detection.from_region(region) for region in regions]
            
            logger.debug(f"Detection complete for {self.input_path}: found {len(detections)} regions")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for {self.input_path}: {e}", exc_info=True)
            raise  # Don't wrap, just propagate original error
        
    async def on_complete(self):
        """Called when processing is complete"""
        logger.info(f"Original image {self.input_path} processing complete")
        
    async def on_crop_complete(self, crop_path: Path):
        """Called when a cropped image signals completion"""
        logger.info(f"Crop {crop_path} from {self.input_path} complete")