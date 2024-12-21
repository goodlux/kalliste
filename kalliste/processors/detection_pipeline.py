"""Detection and tagging pipeline combining YOLO detection with image tagging."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from PIL import Image

from ..detectors.yolo_detector import YOLODetector
from ..detectors.base import Region, DetectionConfig
from .taggers import ImageTagger, get_default_device
from .region_processor import RegionProcessor  # Import RegionProcessor

logger = logging.getLogger(__name__)

class DetectionPipeline:
    """Combines detection and tagging into a single pipeline."""
    
    def __init__(self, 
                 model: str = "yolov8n",
                 face_model: Optional[str] = None,
                 detection_config: List[DetectionConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize the detection pipeline.
        
        Args:
            model: YOLO model name (e.g., 'yolov8n', 'yolov8s', etc.)
            face_model: Optional face detection model name
            detection_config: List of detection configurations
            device: Optional device specification ('mps', 'cuda', 'cpu', or None)
        """
        self.device = device or get_default_device()
        logger.info(f"Initializing pipeline on device: {self.device}")
        
        self.detector = YOLODetector(
            model=model,
            face_model=face_model,
            config=detection_config
        )
        self.tagger = ImageTagger(device=self.device)
        logger.info("Detection pipeline initialized")
        
    async def process_region(self, 
                           image_path: Path,
                           region: Region,
                           output_dir: Path) -> Dict[str, Any]:
        """Process a single detected region."""
        # Load image for dimensions
        image = Image.open(str(image_path))
        image_size = image.size
        
        # Process region using RegionProcessor
        target_dims = RegionProcessor.get_target_dimensions(
            region.x2 - region.x1,
            region.y2 - region.y1
        )
        
        if target_dims is None:
            return None  # Skip regions that don't meet SDXL requirements
            
        adjusted_region = RegionProcessor.expand_bbox_to_ratio(
            bbox=region,
            target_dims=target_dims,
            original_size=image_size,
            region_type=region.region_type
        )
        
        if adjusted_region is None:
            return None
            
        # Crop using adjusted region
        crop = image.crop((
            adjusted_region.x1,
            adjusted_region.y1,
            adjusted_region.x2,
            adjusted_region.y2
        ))
        
        # Save crop
        crop_filename = f"{image_path.stem}_{adjusted_region.region_type}_{adjusted_region.x1}_{adjusted_region.y1}.jpg"
        crop_path = output_dir / crop_filename
        crop.save(crop_path, "JPEG", quality=95)
        
        # Get tags for crop
        tags = await self.tagger.tag_image(crop_path)
        
        return {
            'region': adjusted_region.__dict__,
            'tags': tags,
            'crop_path': str(crop_path)
        }
    
    async def process_image(self, 
                          image_path: Path,
                          output_dir: Path) -> List[Dict[str, Any]]:
        """
        Process a single image through the detection pipeline.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save processed crops
            
        Returns:
            List of dictionaries containing detection regions and associated tags
        """
        # Ensure paths are Path objects
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run detection
        regions = self.detector.detect(image_path)
        
        # Process each region
        tasks = [
            self.process_region(image_path, region, output_dir)
            for region in regions
        ]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (regions that didn't meet SDXL requirements)
        return [r for r in results if r is not None]