"""YOLO detector implementation."""
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from .base import BaseDetector, Region
from .yolo_classes import YOLO_CLASSES, CLASS_GROUPS
from ..model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """YOLO-based object detector."""
    
    def __init__(self, config: Dict):
        """Initialize YOLO detector with config."""
        super().__init__(config)
        self.model = ModelRegistry.get_model("yolo")["model"]
        # Ensure model is in inference mode
        self.model.eval()
    
    def detect(self, 
              image_path: Path, 
              detection_types: List[str],
              config: Dict) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            detection_types: List of types to detect
            config: Configuration for each detection type
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            results = []
            
            # Convert detection types to YOLO class indices
            class_ids = set()
            for det_type in detection_types:
                if det_type in YOLO_CLASSES:
                    class_ids.add(YOLO_CLASSES[det_type])
                elif det_type in CLASS_GROUPS:
                    for class_name in CLASS_GROUPS[det_type]:
                        class_ids.add(YOLO_CLASSES[class_name])
            
            # Get the highest confidence and lowest IOU threshold across all types
            confidence = min(cfg['confidence'] for cfg in config.values())
            iou_threshold = min(cfg['iou_threshold'] for cfg in config.values())
            
            # Run inference
            pred = self.model(
                str(image_path),
                conf=confidence,
                iou=iou_threshold,
                classes=list(class_ids)  # Filter for requested classes
            )[0]
            
            # Convert to Region objects
            for result in pred.boxes:
                cls_id = int(result.cls[0])
                det_type = next(
                    k for k, v in YOLO_CLASSES.items() 
                    if v == cls_id and k in detection_types
                )
                xyxy = result.xyxy[0].cpu().numpy()
                region = Region(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    region_type=det_type,
                    confidence=float(result.conf[0])
                )
                results.append(region)
            
            return results
            
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise