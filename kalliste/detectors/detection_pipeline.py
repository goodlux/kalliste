"""Detection pipeline for coordinating different detectors and configurations."""
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from .base import Region
from .yolo_detector import YOLODetector
from .yolo_classes import YOLO_CLASSES, CLASS_GROUPS

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results."""
    regions: List[Region]
    detector_used: str
    model_identifier: str
    detection_types: List[str]

class DetectionPipeline:
    """Coordinates detection process based on configuration."""
    
    def detect(self, image_path: Path, config: Dict) -> DetectionResult:
        """Run detection on an image using configuration.
        
        Args:
            image_path: Path to image file
            config: Detection configuration from batch
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Get detection types from config
            detection_types = list(config.keys() - {'target', 'device'})
            logger.debug(f"Detection types from config: {detection_types}")
            
            # Check if detection types are YOLO-supported
            yolo_types = []
            for det_type in detection_types:
                # Check direct classes
                if det_type in YOLO_CLASSES:
                    yolo_types.append(det_type)
                # Check group classes
                elif det_type in CLASS_GROUPS:
                    yolo_types.extend(CLASS_GROUPS[det_type])
                    
            if not yolo_types:
                raise ValueError(f"No supported detection types found in: {detection_types}")
                
            # Get detector settings for each type
            detector_config = {
                dtype: {
                    'confidence': config[dtype].get('confidence', 0.5),
                    'iou_threshold': config[dtype].get('iou_threshold', 0.45)
                }
                for dtype in yolo_types
            }
            logger.debug(f"Detector config: {detector_config}")
            
            # Initialize YOLO detector
            detector = YOLODetector(config)
            
            # Run detection with config
            regions = detector.detect(
                image_path=image_path,
                detection_types=yolo_types,
                config=detector_config
            )
            
            return DetectionResult(
                regions=regions,
                detector_used='yolo',
                model_identifier='yolov11x',
                detection_types=yolo_types
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}") from e