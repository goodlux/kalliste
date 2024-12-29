"""Detection pipeline for coordinating different detectors and configurations."""
from pathlib import Path
from typing import List, Dict
import logging
from dataclasses import dataclass

from .base import Region
from .yolo_detector import YOLODetector
from .yolo_face_detector import YOLOFaceDetector
from .yolo_classes import YOLO_CLASSES

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results."""
    regions: List[Region]
    detection_types: List[str]

class DetectionPipeline:
    """Coordinates detection process based on configuration."""
    
    def detect(self, image_path: Path, config: Dict) -> DetectionResult:
        """Run detection on an image using configuration.
        
        Args:
            image_path: Path to image file
            config: Detection configuration with detection types and thresholds
                   e.g. {
                       'face': {'confidence_threshold': 0.6, 'nms_threshold': 0.7},
                       'person': {'confidence_threshold': 0.5, 'nms_threshold': 0.7}
                   }
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            results = []
            detection_types = list(config.keys())
            logger.debug(f"Requested detection types: {detection_types}")
            
            # Handle face detection if requested
            if 'face' in detection_types:
                face_detector = YOLOFaceDetector(config)
                face_config = config['face']
                results.extend(face_detector.detect(
                    image_path=image_path,
                    confidence_threshold=face_config.get('confidence_threshold', YOLOFaceDetector.DEFAULT_CONFIDENCE_THRESHOLD),
                    nms_threshold=face_config.get('nms_threshold', YOLOFaceDetector.DEFAULT_NMS_THRESHOLD)
                ))
                detection_types.remove('face')
            
            # Handle YOLO detections for remaining types
            yolo_types = [t for t in detection_types if t in YOLO_CLASSES]
            if yolo_types:
                yolo_detector = YOLODetector(config)
                # Convert types to class IDs
                class_ids = [YOLO_CLASSES[t] for t in yolo_types]
                
                # Get lowest thresholds from any type's config
                confidence_threshold = min(
                    config[t].get('confidence_threshold', YOLODetector.DEFAULT_CONFIDENCE_THRESHOLD)
                    for t in yolo_types
                )
                nms_threshold = min(
                    config[t].get('nms_threshold', YOLODetector.DEFAULT_NMS_THRESHOLD) 
                    for t in yolo_types
                )
                
                # Run YOLO detection
                yolo_regions = yolo_detector.detect(
                    image_path=image_path,
                    classes=class_ids,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold
                )
                
                # Map class IDs back to type names
                class_id_to_type = {v: k for k, v in YOLO_CLASSES.items()}
                for region in yolo_regions:
                    region.region_type = class_id_to_type[int(region.region_type)]
                
                results.extend(yolo_regions)
            
            # Warn about any unsupported types
            unsupported = set(detection_types) - {'face'} - set(YOLO_CLASSES.keys())
            if unsupported:
                logger.warning(f"Unsupported detection types: {unsupported}")
            
            return DetectionResult(
                regions=results,
                detection_types=[r.region_type for r in results]
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}") from e