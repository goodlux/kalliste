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
    KallisteBagTag,
    KallisteIntegerTag
)
import datetime

logger = logging.getLogger(__name__)

class OriginalImage:
    """Processes original images and creates derived crops."""
    
    def __init__(self, source_path: Path, output_dir: Path, config: Dict):
        self.source_path = source_path
        self.output_dir = output_dir
        self.config = config


    def _extract_lr_metadata(self) -> tuple[Dict[str, any], List[Dict[str, any]], List[str]]:
        """Extract all Lightroom metadata using a single exiftool call."""
        try:
            cmd = [
                'exiftool',
                '-RegionAppliedToDimensionsW',
                '-RegionAppliedToDimensionsH',
                '-RegionName',
                '-RegionType', 
                '-RegionAreaH',
                '-RegionAreaW',
                '-RegionAreaX',
                '-RegionAreaY',
                '-RegionRotation',
                '-WeightedFlatSubject',
                '-Rating',           # Added Rating
                '-Label',            # Added Label
                '-j',
                '-G',
                str(self.source_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Exiftool failed: {result.stderr}")
                return {}, [], []
                
            metadata = json.loads(result.stdout)[0]
            logger.debug(f"Raw exiftool output: {json.dumps(metadata, indent=2)}")
        
            # Extract general metadata
            general_metadata = {
                'photoshoot_id': self.source_path.parent.name,
            }
            
            # Add Rating if present
            if 'Rating' in metadata:
                general_metadata['lr_rating'] = metadata['Rating']
                
            # Add Label if present (convert to lowercase)
            if 'Label' in metadata:
                general_metadata['lr_label'] = metadata['Label'].lower()
            
            # Process LR subject tags from WeightedFlatSubject
            lr_tags = []
            weighted_subject = metadata.get('XMP:WeightedFlatSubject', [])
            if isinstance(weighted_subject, list):
                lr_tags = weighted_subject
            elif isinstance(weighted_subject, str):
                lr_tags = [tag.strip() for tag in weighted_subject.split(',')]
            
            # Process the LR face regions
            lr_faces = []

            # Get all the region arrays - updated to use correct keys and handle float values
            names = metadata.get('XMP:RegionName', [])  # Already comes as a list
            if not isinstance(names, list):
                names = [n.strip() for n in str(names).split(',')]
                
            types = metadata.get('XMP:RegionType', [])
            if not isinstance(types, list):
                types = [t.strip() for t in str(types).split(',')]
                
            img_width = float(metadata.get('XMP:RegionAppliedToDimensionsW', 0))
            img_height = float(metadata.get('XMP:RegionAppliedToDimensionsH', 0))

            # Helper function to convert value to float list
            def to_float_list(value):
                if isinstance(value, list):
                    return [float(x) for x in value]
                elif isinstance(value, (int, float)):
                    return [float(value)]
                else:
                    return [float(x) for x in str(value).split(',')]

            areas_h = to_float_list(metadata.get('XMP:RegionAreaH', []))
            areas_w = to_float_list(metadata.get('XMP:RegionAreaW', []))
            areas_x = to_float_list(metadata.get('XMP:RegionAreaX', []))
            areas_y = to_float_list(metadata.get('XMP:RegionAreaY', []))

            logger.debug(f"Processing regions: names={names}, types={types}")
            
            for i in range(len(names)):
                if types[i].lower() == 'face':
                    x_center = areas_x[i] * img_width
                    y_center = areas_y[i] * img_height
                    width = areas_w[i] * img_width
                    height = areas_h[i] * img_height
                    
                    lr_faces.append({
                        'name': names[i],
                        'bbox': {
                            'x': x_center - (width/2),
                            'y': y_center - (height/2),
                            'w': width,
                            'h': height
                        }
                    })
                    logger.debug(f"Added face record for {names[i]}: center=({x_center},{y_center}), size=({width},{height})")
                
            return general_metadata, lr_faces, lr_tags
                
        except Exception as e:
            logger.error(f"Failed to extract Lightroom metadata: {e}", exc_info=True)
            return {}, [], []

    async def process(self):
        """Process the image."""
        try:
            # Get all Lightroom metadata in one call
            general_metadata, lr_faces, lr_tags = self._extract_lr_metadata()
            logger.debug(f"Extracted general metadata: {general_metadata}")
            logger.debug(f"Extracted faces: {lr_faces}")
            logger.debug(f"Extracted tags: {lr_tags}")

            # Pass relevant config to detection pipeline
            detection_pipeline = DetectionPipeline()
            results = detection_pipeline.detect(
                self.source_path,
                config=self.config['detector']
            )
            
            logger.debug(f"About to match regions. Have lr_faces: {lr_faces}")
            # If we have LR faces, try to match them
            if lr_faces:
                region_matcher = RegionMatcher()
                results.regions = region_matcher.match_faces(results.regions, lr_faces)
            
            # Process each detected region
            for region in results.regions:
                # Add basic metadata tags
                if general_metadata.get('datetime_original'):
                    region.add_tag(KallisteDateTag(
                        "KallistePhotoshootDate",
                        datetime.strptime(general_metadata['datetime_original'], '%Y:%m:%d %H:%M:%S')
                    ))
                
                if general_metadata.get('rating'):
                    region.add_tag(KallisteIntegerTag(
                        "KallisteRating", 
                        int(general_metadata['rating'])
                    ))
                
                region.add_tag(KallisteStringTag(
                    "KallistePhotoshootId", 
                    general_metadata['photoshoot_id']
                ))
                
                region.add_tag(KallisteStringTag(
                    "KallisteOriginalFilePath", 
                    str(self.source_path.absolute())
                ))
                
                # Add Lightroom tags if any exist
                if lr_tags:
                    logger.debug(f"LR tags before creating bag tag: {lr_tags}")
                    region.add_tag(KallisteBagTag("KallisteLrTags", set(lr_tags)))
                
                # Create and process cropped image
                cropped = CroppedImage(
                    self.source_path,
                    self.output_dir,
                    region,
                    config=self.config
                )
                await cropped.process()

                # Add Rating if present
                if 'lr_rating' in general_metadata:
                    region.add_tag(KallisteIntegerTag(
                        "KallisteLrRating", 
                        int(general_metadata['lr_rating'])
                    ))
                
                # Add Label if present
                if 'lr_label' in general_metadata:
                    region.add_tag(KallisteStringTag(
                        "KallisteLrLabel", 
                        general_metadata['lr_label']
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to process image {self.source_path}: {e}")
            raise