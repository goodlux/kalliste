"""YOLO-based detector for standard object detection."""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
import logging

from .base import BaseDetector, DetectionConfig, Region
from ..config import YOLO_CACHE_DIR, YOLO_PERSON_MODEL

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """YOLO-based detector for standard object detection."""
    
    def __init__(self, config: List[DetectionConfig]):
        """Initialize YOLO detector.
        
        Args:
            config: List of detection configurations
        """
        super().__init__(config)
        logger.info("Initializing YOLO detector")
        
        self.model_path = YOLO_CACHE_DIR / YOLO_PERSON_MODEL
        self._model_fused = False
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
            
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(str(self.model_path), task='detect')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}", exc_info=True)
            raise
            
        # Map YOLO class indices to our detection types
        # Standard YOLO COCO classes
        self.class_map = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            15: 'cat',
            16: 'dog',
            # Add more as needed
        }
        
        # Filter class map to only include configured detection types
        self.active_classes = {
            class_id: det_type 
            for class_id, det_type in self.class_map.items()
            if det_type in self.config
        }
        
        logger.info(f"YOLODetector initialized with {len(self.active_classes)} active classes")
    
    def _ensure_model_fused(self):
        """Ensure model is fused, handling errors gracefully."""
        if not self._model_fused:
            try:
                logger.info("Fusing model...")
                self.model.model.fuse()
                self._model_fused = True
                logger.info("Model fused successfully")
            except Exception as e:
                logger.warning(f"Model fusion failed: {str(e)}")
                logger.warning("Continuing with unfused model")
    
    def get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        with Image.open(image_path) as img:
            return img.size
    
    def detect(self, image_path: Path) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected regions meeting configuration requirements
        """
        logger.info(f"Running detection on {image_path}")
        regions = []
        image_size = self.get_image_size(image_path)
        
        try:
            # Skip if no active classes
            if len(self.active_classes) == 0:
                return regions
                
            # Ensure model is fused
            self._ensure_model_fused()
            
            # Get minimum confidence threshold across all active detection types
            min_conf = min(self.config[det_type].confidence_threshold 
                          for det_type in self.active_classes.values())
            
            # Run YOLO detection
            logger.debug("Running YOLO detection")
            results = self.model(str(image_path), conf=min_conf)[0]
            
            # Process each detection
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    # Skip if not a class we're looking for
                    if class_id not in self.active_classes:
                        continue
                        
                    det_type = self.active_classes[class_id]
                    conf = float(box.conf[0])
                    
                    # Skip if below type-specific confidence threshold
                    if conf < self.config[det_type].confidence_threshold:
                        continue
                    
                    # Create region from detection
                    xyxy = box.xyxy[0].cpu().numpy()
                    region = Region(
                        x1=int(xyxy[0]), 
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]), 
                        y2=int(xyxy[3]),
                        region_type=det_type,
                        confidence=conf
                    )
                    
                    # Adjust region to meet SDXL requirements
                    adjusted_region = self.adjust_for_sdxl(region, image_size)
                    if adjusted_region:
                        regions.append(adjusted_region)
            
            logger.info(f"Detection complete. Found {len(regions)} total regions")
            return regions
            
        except Exception as e:
            logger.error("Detection failed", exc_info=True)
            raise