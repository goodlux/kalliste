"""YOLO face detector implementation."""
from pathlib import Path
from typing import List, Dict
import logging
from .base import BaseDetector
from ..region import Region
from ..model.model_registry import ModelRegistry
from ..tag.kalliste_tag import KallisteStructureTag  # Add this import

logger = logging.getLogger(__name__)

class YOLOFaceDetector(BaseDetector):
    """YOLO-based face detector."""
    
    # Default thresholds for face detection
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_NMS_THRESHOLD = 0.7
    
    def __init__(self, config: Dict):
        """Initialize YOLO face detector with config."""
        super().__init__(config)
        self.model = ModelRegistry.get_model("yolo-face")["model"]
        self.model.eval()
    
    def detect(self, 
              image_path: Path,
              confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
              nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> List[Region]:
        """Run face detection on an image."""
        self._validate_image_path(image_path)
        
        try:
            # Run inference with face detection model
            pred = self.model(
                str(image_path),
                conf=confidence_threshold,
                iou=nms_threshold
            )[0]
            
            # Convert predictions to Region objects
            regions = []
            for result in pred.boxes:
                xyxy = result.xyxy[0].cpu().numpy()
                confidence = float(result.conf[0])
                
                region = Region(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    region_type='face',
                    confidence=confidence
                )
                
                # Add structured detection data
                detection_data = KallisteStructureTag(
                    "KallisteDetectionData",
                    {
                        "detector": "YOLOv11m-face",
                        "bbox": {
                            "x": int(xyxy[0]),
                            "y": int(xyxy[1]),
                            "width": int(xyxy[2] - xyxy[0]),
                            "height": int(xyxy[3] - xyxy[1])
                        },
                        "confidence": confidence,
                        "type": "face",
                        "metadata": {
                            "confidence_threshold": confidence_threshold,
                            "nms_threshold": nms_threshold
                        }
                    }
                )
                region.add_tag(detection_data)
                
                regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise