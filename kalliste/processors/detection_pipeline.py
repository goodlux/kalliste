"""
Detection and tagging pipeline combining YOLO detection with image tagging.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from PIL import Image
import json
import torch

from ..detectors.yolo_detector import YOLODetector
from ..detectors.base import Region, DetectionConfig
from .taggers import ImageTagger, get_default_device

logger = logging.getLogger(__name__)

class DetectionPipeline:
    """Combines detection and tagging into a single pipeline."""
    
    def __init__(self, 
                 model_path: str,
                 face_model_path: Optional[str] = None,
                 detection_config: List[DetectionConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize the detection pipeline.
        
        Args:
            model_path: Path to YOLO model
            face_model_path: Optional path to face detection model
            detection_config: List of detection configurations
            device: Optional device specification ('mps', 'cuda', 'cpu', or None)
        """
        self.device = device or get_default_device()
        logger.info(f"Initializing pipeline on device: {self.device}")
        
        self.detector = YOLODetector(
            model_path=model_path,
            face_model_path=face_model_path,
            config=detection_config
        )
        self.tagger = ImageTagger(device=self.device)
        logger.info("Detection pipeline initialized")

    # ... [rest of the implementation remains the same] ...
