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
        model_info = ModelRegistry.get_model("yolo-face")
        self.model = model_info["model"]
        self.model.eval()
        
        # Debug: Check if model has CLIP-related attributes
        logger.debug(f"🔍 Inspecting YOLO face model attributes...")
        for attr in dir(self.model):
            if 'clip' in attr.lower() or 'vit' in attr.lower():
                logger.debug(f"   Found potential CLIP attribute: {attr}")
        
        # Check if model has any submodules
        if hasattr(self.model, 'model'):
            logger.debug(f"   Model has nested 'model' attribute")
            if hasattr(self.model.model, 'names'):
                logger.debug(f"   Model names: {getattr(self.model.model, 'names', {})}")
    
    def detect(self, 
              image_path: Path,
              confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
              nms_threshold: float = DEFAULT_NMS_THRESHOLD) -> List[Region]:
        """Run face detection on an image."""
        self._validate_image_path(image_path)
        
        try:
            logger.debug(f"🔍 Running YOLO face detection on {image_path.name}")
            logger.debug(f"   Model type: {type(self.model)}")
            logger.debug(f"   Model device: {getattr(self.model, 'device', 'unknown')}")
            
            # Check model internals before running
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
                logger.debug(f"   Model has nested structure")
                inner_model = self.model.model.model
                if hasattr(inner_model, 'names'):
                    logger.debug(f"   Model names: {inner_model.names}")
            
            logger.debug("🎯 About to call model() for inference")
            # Run inference with face detection model
            pred = self.model(
                str(image_path),
                conf=confidence_threshold,
                iou=nms_threshold,
                verbose=False  # Suppress YOLO's output
            )[0]
            logger.debug("✔️ model() inference complete")
            
            logger.debug(f"✅ YOLO face detection completed, found {len(pred.boxes)} faces")
            
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