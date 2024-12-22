from pathlib import Path
from typing import List, Optional
from .types import Detection, ProcessingStatus
from .cropped_image import CroppedImage
from ..detectors.base import DetectionConfig
from ..detectors.yolo_detector import YOLODetector
import asyncio

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
    
    def __init__(self, path: Path):
        self.path = path
        self.metadata = {}  # Original metadata
        self.cropped_images: List[CroppedImage] = []
        self.status = ProcessingStatus.PENDING
        
        # Initialize detector
        self.detector = YOLODetector(
            model="yolov8n",  # Use nano model by default
            face_model="yolov8n-face",  # Add face detection
            config=self.DEFAULT_CONFIGS
        )
        
    async def process(self):
        """Main processing pipeline"""
        self.status = ProcessingStatus.PROCESSING
        
        try:
            # Run detections
            detections = await self.run_detectors()
            
            if detections:
                # Create and process crops asynchronously
                crop_tasks = []
                for detection in detections:
                    cropped = CroppedImage(
                        source_path=self.path,
                        detection=detection
                    )
                    self.cropped_images.append(cropped)
                    crop_tasks.append(cropped.process())
                    
                # Wait for all crops to complete
                await asyncio.gather(*crop_tasks)
                
            self.status = ProcessingStatus.COMPLETE
            # Signal completion
            await self.on_complete()
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            print(f"Error processing {self.path}: {e}")
            raise
        
    async def run_detectors(self) -> List[Detection]:
        """Run all detectors on image asynchronously"""
        # Run detector in an executor to not block
        loop = asyncio.get_event_loop()
        regions = await loop.run_in_executor(
            None, 
            self.detector.detect,
            self.path
        )
        
        # Convert regions to our Detection type
        return [Detection.from_region(region) for region in regions]
        
    async def on_complete(self):
        """Called when processing is complete"""
        print(f"Original image {self.path} processing complete")
        
    async def on_crop_complete(self, crop_path: Path):
        """Called when a cropped image signals completion"""
        print(f"Crop {crop_path} from {self.path} complete")