"""YOLO face detector implementation."""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
import logging

from .base import BaseDetector, DetectionConfig, Region
from ..config import YOLO_CACHE_DIR, YOLO_FACE_MODEL

logger = logging.getLogger(__name__)

class YOLOFaceDetector(BaseDetector):
    """YOLO-based face detector."""
    
    def __init__(self, config: List[DetectionConfig]):
        """Initialize YOLO face detector.
        
        Args:
            config: List of detection configurations
        """
        super().__init__(config)
        logger.info("Initializing YOLO face detector")
        
        self.model_path = YOLO_CACHE_DIR / YOLO_FACE_MODEL
        self._model_fused = False
        
        # Load face model
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO face model not found at {self.model_path}")
            
        try:
            logger.info(f"Loading YOLO face model from {self.model_path}")
            self.model = YOLO(str(self.model_path), task='detect')
            logger.info("YOLO face model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO face model: {str(e)}")
            raise
    
    def _ensure_model_fused(self):
        """Ensure model is fused, handling errors gracefully."""
        if not self._model_fused:
            try:
                logger.info("Fusing face model...")
                self.model.model.fuse()
                self._model_fused = True
                logger.info("Face model fused successfully")
            except Exception as e:
                logger.warning(f"Model fusion failed: {str(e)}")
                logger.warning("Continuing with unfused model")
    
    def get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        with Image.open(image_path) as img:
            return img.size
    
    def detect(self, image_path: Path) -> List[Region]:
        """Run face detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected face regions meeting configuration requirements
        """
        logger.info(f"Running face detection on {image_path}")
        regions = []
        image_size = self.get_image_size(image_path)
        
        # Get face-specific config
        face_config = self.config.get('face')
        if not face_config:
            return []
            
        try:
            # Ensure model is fused
            self._ensure_model_fused()
            
            # Run face detection
            results = self.model(str(image_path), conf=face_config.confidence_threshold)[0]
            
            # Process each detection
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    region = Region(
                        x1=int(xyxy[0]), 
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]), 
                        y2=int(xyxy[3]),
                        region_type='face',
                        confidence=conf
                    )
                    
                    # Adjust region to meet SDXL requirements
                    adjusted_region = self.adjust_for_sdxl(region, image_size)
                    if adjusted_region:
                        regions.append(adjusted_region)
                        
            logger.info(f"Face detection complete. Found {len(regions)} faces")
            return regions
            
        except Exception as e:
            logger.error("Face detection failed", exc_info=True)
            raise