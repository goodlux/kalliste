"""YOLO face detector implementation."""
from pathlib import Path
from typing import List, Dict
import logging
from .base import BaseDetector, Region
from ..model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class YOLOFaceDetector(BaseDetector):
    """YOLO-based face detector."""
    
    def __init__(self, config: Dict):
        """Initialize YOLO face detector with config."""
        super().__init__(config)
        self.model = ModelRegistry.get_model("yolo-face")["model"]
        # Ensure model is in inference mode
        self.model.eval()
    
    def detect(self, 
              image_path: Path,
              config: Dict) -> List[Region]:
        """Run face detection on an image.
        
        Args:
            image_path: Path to image file
            config: Configuration for face detection
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            results = []
            
            # Get confidence and IOU settings
            confidence = config.get('confidence', 0.6)
            iou_threshold = config.get('iou_threshold', 0.35)
            
            # Run inference
            pred = self.model(
                str(image_path),
                conf=confidence,
                iou=iou_threshold
            )[0]
            
            # Convert to Region objects
            for result in pred.boxes:
                xyxy = result.xyxy[0].cpu().numpy()
                region = Region(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    region_type='face',
                    confidence=float(result.conf[0])
                )
                results.append(region)
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}", exc_info=True)
            raise