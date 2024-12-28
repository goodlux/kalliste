"""Handles processing of original images and creation of derived crops."""
from pathlib import Path
from typing import List, Optional, Dict
from ..types import  ProcessingStatus
from .cropped_image import CroppedImage
from ..detectors.base import DetectionConfig, Region
from ..detectors.detection_pipeline import DetectionPipeline, DetectionResult
from ..model.model_registry import ModelRegistry
import asyncio
import logging

logger = logging.getLogger(__name__)

class OriginalImage:
    def __init__(self, source_path: Path, output_dir: Path, config: Dict):
        self.source_path = source_path
        self.output_dir = output_dir
        self.config = config
        
    async def process(self):
        """Process the image."""
        # Pass relevant config to detection pipeline
        detection_pipeline = DetectionPipeline()
        results = detection_pipeline.detect(
            self.source_path,
            config=self.config['detector']  # Pass detector config
        )
        
        # Create cropped images with relevant config
        for region in results.regions:
            cropped = CroppedImage(
                self.source_path,
                self.output_dir,
                region,
                config=self.config  # Pass full config for tagger access
            )
            await cropped.process()