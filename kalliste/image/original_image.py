"""Handles processing of original images and creation of derived crops."""
from pathlib import Path
from typing import List, Optional, Dict, Set
import asyncio
import logging
import subprocess
import json
from collections import defaultdict
from datetime import datetime
import re

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
                '-XMP:Rating',
                '-XMP:Label',
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
            if 'XMP:Rating' in metadata:
                rating = metadata['XMP:Rating']
                if rating is not None:
                    general_metadata['lr_rating'] = int(rating)
                    logger.debug(f"Found LR rating: {rating}")
                
            # Add Label if present (convert to lowercase)
            if 'XMP:Label' in metadata:
                label = metadata['XMP:Label']
                if label:
                    general_metadata['lr_label'] = label.lower()
                    logger.debug(f"Found LR label: {label}")
            
            # Process LR subject tags
            lr_tags = []
            weighted_subject = metadata.get('XMP:WeightedFlatSubject', [])
            if isinstance(weighted_subject, list):
                lr_tags = weighted_subject
            elif isinstance(weighted_subject, str):
                lr_tags = [tag.strip() for tag in weighted_subject.split(',')]
            
            # Process the LR face regions
            lr_faces = []

            # Get all the region arrays
            names = metadata.get('XMP:RegionName', [])
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
                    logger.debug(f"Added face record for {names[i]}")
                
            return general_metadata, lr_faces, lr_tags
                
        except Exception as e:
            logger.error(f"Failed to extract Lightroom metadata: {e}", exc_info=True)
            return {}, [], []

    def _extract_path_metadata(self, region: Region):
        """Extract and set metadata tags directly from the path."""
        try:
            folder_name = self.source_path.parent.name
            parts = folder_name.split('_')
            
            # Try to parse date from first part
            if parts and re.match(r'^\d{8}$', parts[0]):
                try:
                    date = datetime.strptime(parts[0], '%Y%m%d')
                    region.add_tag(KallisteDateTag("KallistePhotoshootDate", date))
                    # After date, first part before any @ or # is the shoot name
                    if len(parts) > 1:
                        shoot_name = next((part for part in parts[1:] 
                                        if not part.startswith('@') and 
                                        not part.startswith('#')), None)
                        if shoot_name:
                            region.add_tag(KallisteStringTag(
                                "KallistePhotoshootName",
                                shoot_name
                            ))
                except ValueError:
                    pass

            # Process each part
            for part in parts:
                if not part:
                    continue
                    
                # Person name starts with @
                if part.startswith('@'):
                    region.add_tag(KallisteStringTag(
                        "KallistePersonName",
                        part[1:]  # Remove @ symbol
                    ))
                    
                # Source type starts with #    
                elif part.startswith('#'):
                    region.add_tag(KallisteStringTag(
                        "KallisteSourceType",
                        part[1:]  # Remove # symbol
                    ))
                
                # Everything else goes to additional tags
                elif not re.match(r'^\d{8}$', part):  # Skip the date part
                    # We'll collect these and add as a bag tag at the end
                    if not hasattr(self, '_additional_tags'):
                        self._additional_tags = set()
                    self._additional_tags.add(part)
                    
            # Add collected tags as a bag tag
            if hasattr(self, '_additional_tags'):
                region.add_tag(KallisteBagTag(
                    "KallisteTags", 
                    self._additional_tags
                ))
                delattr(self, '_additional_tags')
                
        except Exception as e:
            logger.error(f"Failed to extract path metadata: {e}", exc_info=True)

    async def process(self) -> Dict[str, Dict]:
        """
        Process the image and return statistics.
        Returns:
            Dict: Statistics organized by region type, including:
                - count of regions detected
                - count of regions rejected for size
                - assessment counts (technical, aesthetic, overall, kalliste)
        """
        try:
            # Initialize statistics structure
            stats = defaultdict(lambda: {
                'count': 0,
                'rejected_small': 0,
                'technical': defaultdict(int),
                'aesthetic': defaultdict(int),
                'overall': defaultdict(int),
                'kalliste': defaultdict(int)
            })

            # Get all Lightroom metadata in one call
            general_metadata, lr_faces, lr_tags = self._extract_lr_metadata()
            logger.debug(f"Extracted LR metadata: {general_metadata}")
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
                # Record region detection in statistics
                region_type = region.region_type
                stats[region_type]['count'] += 1

                # Extract and set path-based metadata tags directly
                self._extract_path_metadata(region)

                # Add LR-based metadata tags
                if general_metadata.get('datetime_original'):
                    region.add_tag(KallisteDateTag(
                        "KallisteDateTimeOriginal",
                        datetime.strptime(general_metadata['datetime_original'], '%Y:%m:%d %H:%M:%S')
                    ))
                
                if general_metadata.get('rating'):
                    region.add_tag(KallisteIntegerTag(
                        "KallisteRating", 
                        int(general_metadata['rating'])
                    ))
                
                # Update in process() method, after extracting path metadata:
                # Remove this line since we'll create KallistePhotoshootId directly from the folder name parts
                # region.add_tag(KallisteStringTag("KallistePhotoshootId", general_metadata['photoshoot_id']))
                
                region.add_tag(KallisteStringTag(
                    "KallisteOriginalFilePath", 
                    str(self.source_path.absolute())
                ))

                # Add training target from config
                target_platform = self.config.get('target', {}).get('platform', 'SDXL')
                region.add_tag(KallisteStringTag(
                    "KallisteTrainingTarget",
                    target_platform
                ))
                
                # Add Lightroom tags if any exist
                if lr_tags:
                    logger.debug(f"LR tags before creating bag tag: {lr_tags}")
                    region.add_tag(KallisteBagTag("KallisteLrTags", set(lr_tags)))
                
                # Add Rating if present
                if 'lr_rating' in general_metadata:
                    star_rating = general_metadata['lr_rating']
                    rating_str = 'unrated' if star_rating == 0 else f"{star_rating}_star"
                    region.add_tag(KallisteStringTag(
                        "KallisteLrRating", 
                        rating_str
                    ))
                    logger.debug(f"Added LR rating tag: {rating_str}")
                
                # Add Label if present
                if 'lr_label' in general_metadata:
                    region.add_tag(KallisteStringTag(
                        "KallisteLrLabel", 
                        general_metadata['lr_label']
                    ))
                    logger.debug(f"Added LR label tag: {general_metadata['lr_label']}")
                
                # Process the region and get statistics
                cropped = CroppedImage(
                    self.source_path,
                    self.output_dir,
                    region,
                    config=self.config
                )
                crop_stats = await cropped.process()
                
                # Update statistics from crop processing
                if crop_stats:
                    if crop_stats['rejected_small']:
                        stats[region_type]['rejected_small'] += 1
                    if crop_stats['technical']:
                        stats[region_type]['technical'][crop_stats['technical']] += 1
                    if crop_stats['aesthetic']:
                        stats[region_type]['aesthetic'][crop_stats['aesthetic']] += 1
                    if crop_stats['overall']:
                        stats[region_type]['overall'][crop_stats['overall']] += 1
                    if crop_stats['kalliste']:
                        stats[region_type]['kalliste'][crop_stats['kalliste']] += 1
            
            return dict(stats)  # Convert defaultdict to regular dict
                    
        except Exception as e:
            logger.error(f"Failed to process image {self.source_path}: {e}")
            raise