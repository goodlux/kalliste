"""Handles processing of original images and creation of derived crops."""
from pathlib import Path
from typing import List, Optional, Dict, Set
import asyncio
import logging
import subprocess
import json

from ..types import ProcessingStatus
from .cropped_image import CroppedImage
from ..region import Region, RegionMatcher
from ..detectors.detection_pipeline import DetectionPipeline, DetectionResult
from ..model.model_registry import ModelRegistry
from ..tag.kalliste_tag import (
    KallisteStringTag,
    KallisteDateTag,
    KallisteRealTag,
    KallisteBagTag
)

logger = logging.getLogger(__name__)

class OriginalImage:
    """Processes original images and creates derived crops."""
    
    def __init__(self, source_path: Path, output_dir: Path, config: Dict):
        self.source_path = source_path
        self.output_dir = output_dir
        self.config = config

    def _extract_lr_metadata(self) -> tuple[List[Dict[str, any]], List[str]]:
        """Extract all Lightroom metadata (faces and tags) using a single exiftool call."""
        try:
            # Run exiftool to get both region data and subject tags - add -G for groups
            cmd = [
                'exiftool',
                '-RegionInfo',           # Try this instead of Region:all
                '-RegionAppliedToDimensionsW',
                '-RegionAppliedToDimensionsH',
                '-RegionName',
                '-RegionType',
                '-RegionArea',
                '-WeightedFlatSubject',
                '-j',                    # Output as JSON
                '-G',                    # Show group names
                str(self.source_path)
            ]
            
            logger.debug(f"Running exiftool command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Exiftool failed: {result.stderr}")
                return [], []
                
            # Log raw output to see what we're getting
            logger.debug(f"Exiftool raw output: {result.stdout}")
                    
            # Parse the metadata
            metadata = json.loads(result.stdout)[0]  # First (only) image
            logger.debug(f"Raw exiftool output: {json.dumps(metadata, indent=2)}")
        
            # Check what fields we actually got
            logger.debug(f"Available metadata fields: {list(metadata.keys())}")
                
            # Process the LR face regions, will be used to match Yolo Face detections. 
            face_records = []
            if 'Region Name' in metadata:
                names = metadata.get('Region Name', '').split(', ')
                types = metadata.get('Region Type', '').split(', ')
                rotations = [float(x) for x in metadata.get('Region Rotation', '0').split(', ')]
                img_width = float(metadata.get('Region Applied To Dimensions W', 0))
                img_height = float(metadata.get('Region Applied To Dimensions H', 0))
                areas_h = [float(x) for x in metadata.get('Region Area H', '0').split(', ')]
                areas_w = [float(x) for x in metadata.get('Region Area W', '0').split(', ')]
                areas_x = [float(x) for x in metadata.get('Region Area X', '0').split(', ')]
                areas_y = [float(x) for x in metadata.get('Region Area Y', '0').split(', ')]
                
                for i in range(len(names)):
                    if types[i].lower() == 'face':
                        face_records.append({
                            'name': names[i],
                            'rotation': rotations[i],
                            'bbox': {
                                'x': areas_x[i] * img_width,
                                'y': areas_y[i] * img_height,
                                'w': areas_w[i] * img_width,
                                'h': areas_h[i] * img_height
                            }
                        })
            else:
                logger.info("No Lightroom face regions found")
            # Process LR subject tags
            lr_tags_str = metadata.get('WeightedFlatSubject', '')
            lr_tags = [tag.strip() for tag in lr_tags_str.split(',')] if lr_tags_str else []
            if not lr_tags:
                logger.info("No Lightroom tags found")
                
            return face_records, lr_tags
                
        except Exception as e:
            logger.error(f"Failed to extract Lightroom metadata: {e}")
            return [], []

    async def process(self):
        """Process the image."""
        try:
            # Get all Lightroom metadata in one call
            lr_faces, lr_tags = self._extract_lr_metadata()
            logger.debug(f"Extracted faces: {lr_faces}")  # Add this
            logger.debug(f"Extracted tags: {lr_tags}")    # Add this

            # Pass relevant config to detection pipeline
            detection_pipeline = DetectionPipeline()
            results = detection_pipeline.detect(
                self.source_path,
                config=self.config['detector']
            )
            
            # If we have LR faces, try to match them
            if lr_faces:
                region_matcher = RegionMatcher()
                results.regions = region_matcher.match_faces(results.regions, lr_faces)
            
            # Process each detected region
            for region in results.regions:
                # Add basic metadata tags directly
                region.add_tag(KallisteStringTag(
                    "KallistePhotoshootId", 
                    self.source_path.parent.name
                ))
                
                region.add_tag(KallisteStringTag(
                    "KallisteOriginalFilePath", 
                    str(self.source_path.absolute())
                ))
                
                region.add_tag(KallisteStringTag(
                    "KallisteRegionType",
                    region.region_type
                ))
                
                # Add Lightroom tags if any exist
                if lr_tags:
                    region.add_tag(KallisteBagTag("KallisteLrTags", set(lr_tags)))
                
                # Create and process cropped image
                cropped = CroppedImage(
                    self.source_path,
                    self.output_dir,
                    region,
                    config=self.config
                )
                await cropped.process()
                
        except Exception as e:
            logger.error(f"Failed to process image {self.source_path}: {e}")
            raise