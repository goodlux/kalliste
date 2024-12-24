"""YOLO-based detector supporting multiple detection types."""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys
import logging

from .base import BaseDetector, DetectionConfig, Region
from ..config import YOLO_CACHE_DIR, YOLO_PERSON_MODEL, YOLO_FACE_MODEL

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """YOLO-based detector supporting multiple detection types."""
    
    def __init__(self, config: List[DetectionConfig]):
        """Initialize YOLO detector.
        
        Args:
            config: List of detection configurations
        """
        super().__init__(config)
        
        logger.info("Initializing YOLO detector")
        
        # Use config paths
        self.model_path = YOLO_CACHE_DIR / YOLO_PERSON_MODEL
        self.face_model_path = YOLO_CACHE_DIR / YOLO_FACE_MODEL
        
        # Track fusion state for each model
        self._model_fused = False
        self._face_model_fused = False
        
        # Load main model
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
            
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(str(self.model_path), task='detect')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}", exc_info=True)
            raise
            
        # Load face model if available
        self.face_model = None
        if self.face_model_path.exists():
            try:
                logger.info(f"Loading YOLO face model from {self.face_model_path}")
                self.face_model = YOLO(str(self.face_model_path), task='detect')
                logger.info("YOLO face model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO face model: {str(e)}", exc_info=True)
                logger.warning("Continuing without face detection")
        else:
            logger.warning(f"Face model not found at {self.face_model_path}")
            logger.warning("Continuing without face detection")
        
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
    
    def _ensure_model_fused(self, model, is_face=False):
        """Ensure model is fused, handling errors gracefully."""
        fused_flag = '_face_model_fused' if is_face else '_model_fused'
        model_name = 'face model' if is_face else 'main model'
        
        if not getattr(self, fused_flag):
            try:
                logger.info(f"Fusing {model_name}...")
                model.model.fuse()
                setattr(self, fused_flag, True)
                logger.info(f"{model_name.title()} fused successfully")
            except Exception as e:
                logger.warning(f"Model fusion failed for {model_name}: {str(e)}")
                logger.warning("Continuing with unfused model")
    
    def get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        with Image.open(image_path) as img:
            return img.size
    
    def detect_faces(self, image_path: Path) -> List[Region]:
        """Run face detection using YOLOv8-face model."""
        if not self.face_model:
            return []
            
        image_size = self.get_image_size(image_path)
        regions = []
        
        # Get face-specific config
        face_config = self.config.get('face')
        if not face_config:
            return []
            
        try:
            # Ensure face model is fused
            self._ensure_model_fused(self.face_model, is_face=True)
            
            # Run face detection
            results = self.face_model(str(image_path), conf=face_config.confidence_threshold)[0]
            
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
                        
            return regions
            
        except Exception as e:
            logger.error("Face detection failed", exc_info=True)
            raise  # Let caller handle the error
    
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
            # First, get face detections if we have a face model
            if 'face' in self.config and self.face_model:
                logger.debug("Running face detection")
                try:
                    face_regions = self.detect_faces(image_path)
                    regions.extend(face_regions)
                    logger.debug(f"Found {len(face_regions)} faces")
                except Exception as e:
                    logger.error(f"Face detection error: {str(e)}")
                    # Continue with other detections even if face detection fails
            
            # Skip other detections if we only want faces
            if len(self.active_classes) == 0:
                return regions
                
            # Ensure main model is fused
            self._ensure_model_fused(self.model)
            
            # Get minimum confidence threshold across all active detection types
            min_conf = min(self.config[det_type].confidence_threshold 
                          for det_type in self.active_classes.values())
            
            # Run YOLO detection
            logger.debug("Running main YOLO detection")
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
            raise  # Let caller handle the error wrapping