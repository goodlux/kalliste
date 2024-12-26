"""Detection pipeline for coordinating different detectors and configurations."""
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import PIL.Image
import PIL.ExifTags
import xml.etree.ElementTree as ET
from PIL.ExifTags import TAGS
import numpy as np

from .base import DetectionConfig, Region
from .yolo_detector import YOLODetector
from .yolo_face_detector import YOLOFaceDetector
from ..image.utils import calculate_iou
from ..config import (
    YOLO_PERSON_MODEL,
    YOLO_FACE_MODEL
)

logger = logging.getLogger(__name__)

# Default detection configuration
DEFAULT_DETECTION_CONFIG = {
    'face': {
        'enabled': True,
        'min_confidence': 0.6,
        'min_size': 96,
        'detect_facial_features': False
    },
    'person': {
        'enabled': True,
        'min_confidence': 0.5,
        'min_size': 128,
        'detect_poses': False
    }
}

@dataclass
class LightroomFaceRegion:
    """Container for face regions from Lightroom metadata."""
    x: float  # Normalized coordinates (0-1)
    y: float
    width: float
    height: float
    name: Optional[str] = None
    tags: Optional[List[str]] = None

@dataclass
class RegionWithTags:
    """Region with associated detection and metadata-based tags."""
    region: Region
    tags: Dict[str, Any]
    lr_match: Optional[LightroomFaceRegion] = None

@dataclass
class DetectionResult:
    """Container for detection results with tags."""
    regions: List[RegionWithTags]
    detector_used: str
    model_identifier: str
    detection_types: List[str]

class DetectionPipeline:
    """Coordinates detection process based on configuration."""
    
    # IoU threshold for matching LR faces with detected faces
    LR_FACE_MATCH_THRESHOLD = 0.5
    
    def __init__(self, model_registry=None):
        """Initialize the detection pipeline.
        
        Args:
            model_registry: Optional ModelRegistry instance for model management
        """
        self.model_registry = model_registry
        self._detectors = {}
        self.exif_metadata = None
        self.xmp_metadata = None
        
    def detect(self, image_path: Path, detection_config: Optional[Dict] = None) -> DetectionResult:
        """Run detection on an image using appropriate detector based on config."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Use default config if none provided
        config = detection_config if detection_config is not None else DEFAULT_DETECTION_CONFIG.copy()
            
        # Load metadata
        self.load_metadata(image_path)
        
        # Get image dimensions for coordinate conversion
        with PIL.Image.open(image_path) as img:
            image_size = img.size
        
        # Get list of detection types needed
        detection_types = [k for k, v in config.items() if v.get('enabled', True)]
        if not detection_types:
            logger.warning("No detection types enabled in config")
            return DetectionResult(
                regions=[],
                detector_used="none",
                model_identifier="none",
                detection_types=[]
            )
        
        try:
            # Determine appropriate detector
            detector_type, model_identifier = self._get_detector_for_types(detection_types)
            
            # Initialize detector if needed
            self._init_detector(detector_type, model_identifier, config)
            
            # Get detector
            detector_key = f"{detector_type}_{model_identifier}"
            detector = self._detectors[detector_key]
            
            # Run detection
            regions = detector.detect(image_path)
            
            # Match regions with metadata (including LR face regions)
            tagged_regions = self.match_lr_faces(regions, image_size)
            
            return DetectionResult(
                regions=tagged_regions,
                detector_used=detector_type,
                model_identifier=model_identifier,
                detection_types=detection_types
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Detection failed: {str(e)}") from e
