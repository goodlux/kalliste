"""Detection pipeline for coordinating different detectors and configurations."""
from pathlib import Path
from typing import List, Dict, Set
import logging
from dataclasses import dataclass

from .base import Region
from .yolo_detector import YOLODetector
from .yolo_face_detector import YOLOFaceDetector
from .yolo_classes import YOLO_CLASSES, CLASS_GROUPS

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results."""
    regions: List[Region]
    detection_types: List[str]

class DetectionPipeline:
    """Coordinates detection process based on configuration."""
    
    def _get_yolo_class_ids(self, detection_types: List[str]) -> Dict[int, str]:
        """Convert detection types to YOLO class IDs and maintain mapping back to type."""
        class_mapping = {}  # maps class_id to detection_type
        for det_type in detection_types:
            if det_type in YOLO_CLASSES:
                class_mapping[YOLO_CLASSES[det_type]] = det_type
            elif det_type in CLASS_GROUPS:
                for class_name in CLASS_GROUPS[det_type]:
                    class_mapping[YOLO_CLASSES[class_name]] = det_type
        return class_mapping
    
    def detect(self, image_path: Path, config: Dict) -> DetectionResult:
        """Run detection on an image using configuration."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Get detection types, ignoring 'target' which is handled elsewhere
            detection_types = [k for k in config.keys() if k != 'target']
            logger.debug(f"Requested detection types: {detection_types}")
            
            results = []
            
            # Handle face detection separately as it uses a specialized model
            if 'face' in detection_types:
                face_detector = YOLOFaceDetector(config)
                results.extend(face_detector.detect(
                    image_path=image_path,
                    config=config.get('face', {})
                ))
                detection_types.remove('face')
            
            # Handle remaining types that YOLO can detect
            yolo_types = [t for t in detection_types if t in YOLO_CLASSES or t in CLASS_GROUPS]
            if yolo_types:
                class_mapping = self._get_yolo_class_ids(yolo_types)
                detector = YOLODetector(config)
                results.extend(detector.detect(
                    image_path=image_path,
                    class_ids=list(class_mapping.keys()),
                    type_mapping=class_mapping,
                    config={t: config[t] for t in yolo_types}
                ))
            
            # Warn about any unsupported types
            unsupported = set(detection_types) - {'face'} - set(YOLO_CLASSES.keys()) - set(CLASS_GROUPS.keys())
            if unsupported:
                logger.warning(f"Unsupported detection types: {unsupported}")
            
            return DetectionResult(
                regions=results,
                detection_types=detection_types
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}") from e