"""YOLO detector implementation."""
from pathlib import Path
from typing import List, Dict
import logging
from ..region import Region
from .base import BaseDetector
from ..region import Region  # Updated import to use canonical Region
from ..model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """YOLO-based object detector."""
    
    # Ultralytics defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_NMS_THRESHOLD = 0.7
    
    def __init__(self, config: Dict):
        """Initialize YOLO detector with config."""
        super().__init__(config)
        model_info = ModelRegistry.get_model("yolo")
        self.model = model_info["model"]
        self.model.eval()
        
        # Debug: Check if model has CLIP-related attributes
        logger.debug(f"🔍 Inspecting regular YOLO model attributes...")
        for attr in dir(self.model):
            if 'clip' in attr.lower() or 'vit' in attr.lower():
                logger.debug(f"   Found potential CLIP attribute: {attr}")
    
    def detect(self, 
              image_path: Path,
              classes: List[int],
              confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
              nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> List[Region]:
        """Run detection on an image.
        
        Args:
            image_path: Path to image file
            classes: List of COCO class IDs to detect
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            List[Region]: List of detected regions with type
        """
        self._validate_image_path(image_path)
        
        try:
            # Run inference with Ultralytics YOLO model
            # Set verbose=False to suppress the "Processing images..." output
            pred = self.model(
                str(image_path),
                conf=confidence_threshold,
                iou=nms_threshold,
                classes=classes,
                verbose=False  # Suppress YOLO's output
            )[0]
            
            # Convert predictions to Region objects
            regions = []
            for result in pred.boxes:
                cls_id = int(result.cls[0])
                xyxy = result.xyxy[0].cpu().numpy()
                
                region = Region(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    region_type=str(cls_id),  # We'll let pipeline map this to readable names
                    confidence=float(result.conf[0])
                )
                regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise