"""Detection pipeline for coordinating different detectors and configurations."""
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
import logging
from dataclasses import dataclass

from .base import DetectionConfig, Region
from ..model.model_registry import ModelRegistry
from ..config import DETECTION_CONFIG

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
            # Get detector from registry with device preference
            detector = ModelRegistry.get_detector(
                device=config.get('device', {}).get('preferred', 'mps')
            )
            
            # Get detection types from config
            detection_types = list(config.keys() - {'target', 'detector', 'tagger', 'device'})
            
            # Get detector settings for each type
            detector_config = {
                dtype: {
                    'confidence': config['detector'][dtype]['confidence'],
                    'iou_threshold': config['detector'][dtype]['iou_threshold']
                }
                for dtype in detection_types
            }
            
            # Run detection with config
            regions = detector.detect(
                image_path=image_path,
                detection_types=detection_types,
                config=detector_config
            )
            
            return DetectionResult(
                regions=regions,
                detector_used="yolo",
                model_identifier="yolov11x",
                detection_types=detection_types
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}") from e