"""YOLO detector implementation."""
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from .base import BaseDetector, Region
from ..config import DETECTION_CONFIG

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """YOLO-based object detector."""
    
    def __init__(self):
        """Initialize YOLO detector."""
        super().__init__()
        self.model = None
        
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
            # Get model from registry if not initialized
            if self.model is None:
                self.model = ModelRegistry.get_detector()
            
            # Run detection with type-specific configs
            results = []
            for det_type in detection_types:
                type_config = config[det_type]
                type_results = self.model(
                    str(image_path),
                    conf=type_config['confidence'],
                    iou=type_config['iou_threshold']
                )[0]
                
                # Convert to Region objects
                for result in type_results.boxes:
                    if result.cls[0] == self.CLASS_MAP[det_type]:  # Map to YOLO class ID
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

    def get_supported_types(self) -> List[str]:
        """Get list of all supported detection types, including groups."""
        types = set(self._yolo_config["types"])
        if "groups" in self._yolo_config:
            types.update(self._yolo_config["groups"].keys())
        return sorted(types)