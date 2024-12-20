"""YOLO-based detector supporting multiple detection types."""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .base import BaseDetector, DetectionConfig, Region

class YOLODetector(BaseDetector):
    """YOLO-based detector supporting multiple detection types."""
    
    def __init__(self, 
                 model_path: str,
                 face_model_path: Optional[str] = None,
                 config: List[DetectionConfig] = None):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model file
            face_model_path: Optional path to YOLOv8-face model
            config: List of detection configurations
        """
        super().__init__(config)
        self.model = YOLO(model_path)
        self.face_model = YOLO(face_model_path) if face_model_path else None
        
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
    
    def detect(self, image_path: Path) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected regions meeting configuration requirements
        """
        regions = []
        image_size = self.get_image_size(image_path)
        
        # First, get face detections if we have a face model
        if 'face' in self.config and self.face_model:
            regions.extend(self.detect_faces(image_path))
        
        # Skip other detections if we only want faces
        if len(self.active_classes) == 0:
            return regions
            
        # Get minimum confidence threshold across all active detection types
        min_conf = min(self.config[det_type].confidence_threshold 
                      for det_type in self.active_classes.values())
        
        # Run YOLO detection
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
        
        return regions