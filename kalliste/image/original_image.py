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

logger = logging.getLogger(__name__)

class OriginalImage:
    """Processes original images and creates derived crops."""
    
    def __init__(self, source_path: Path, output_dir: Path, config: Dict):
        self.source_path = source_path
        self.output_dir = output_dir
        self.config = config

    def _prepare_kalliste_tags(self, region_name: Optional[str] = None) -> Dict[str, Set[str]]:
        """Prepare kalliste tags for the cropped image."""
        kalliste_tags: Dict[str, Set[str]] = {}
        
        # Add person name if we have one
        if region_name:
            kalliste_tags["KallistePersonName"] = {region_name}
        
        # Get photoshoot ID from parent folder name
        photoshoot_id = self.source_path.parent.name
        kalliste_tags["KallistePhotoshootId"] = {photoshoot_id}
        
        # Add original file path
        kalliste_tags["KallisteOriginalFilePath"] = {str(self.source_path.absolute())}
        
        return kalliste_tags

    def _extract_lr_face_metadata(self) -> List[Dict[str, any]]:
        """Extract Lightroom face metadata from image using exiftool."""
        try:
            # Run exiftool to get relevant fields
            cmd = [
                'exiftool',
                '-Region:all',  # Get all region-related fields
                '-j',          # Output as JSON
                '-G',          # Show group names
                str(self.source_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Exiftool failed: {result.stderr}")
                return []
                
            # Parse the metadata into face records
            metadata = json.loads(result.stdout)[0]  # First (only) image
            
            # Check if we have any region data
            if 'RegionName' not in metadata:
                logger.info("No Lightroom face regions found")
                return []
                
            # Get arrays of region data
            names = metadata.get('RegionName', '').split(', ')
            types = metadata.get('RegionType', '').split(', ')
            rotations = [float(x) for x in metadata.get('RegionRotation', '0').split(', ')]
            
            # Get dimension data
            img_width = float(metadata.get('RegionAppliedToDimensionsW', 0))
            img_height = float(metadata.get('RegionAppliedToDimensionsH', 0))
            
            # Get area data as arrays
            areas_h = [float(x) for x in metadata.get('RegionAreaH', '0').split(', ')]
            areas_w = [float(x) for x in metadata.get('RegionAreaW', '0').split(', ')]
            areas_x = [float(x) for x in metadata.get('RegionAreaX', '0').split(', ')]
            areas_y = [float(x) for x in metadata.get('RegionAreaY', '0').split(', ')]
            
            # Build face records
            face_records = []
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
                    
            return face_records
            
        except Exception as e:
            logger.error(f"Failed to extract Lightroom metadata: {e}")
            return []

    async def process(self):
        """Process the image."""
        try:
            # Get Lightroom face metadata
            lr_faces = self._extract_lr_face_metadata()
            
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
                # Note: match_faces adds name attribute to matched regions
            
            # Create base kalliste tags that every cropped image will get
            base_kalliste_tags = {
                "KallistePhotoshootId": {self.source_path.parent.name},
                "KallisteOriginalFilePath": {str(self.source_path.absolute())}
            }
            
            # Process each detected region
            for region in results.regions:
                # Start with base tags
                kalliste_tags = base_kalliste_tags.copy()
                
                # Add person name if this region was matched
                if hasattr(region, 'name') and region.name:
                    kalliste_tags["KallistePersonName"] = {region.name}
                
                # Create and process cropped image
                cropped = CroppedImage(
                    self.source_path,
                    self.output_dir,
                    region,
                    config=self.config,
                    kalliste_tags=kalliste_tags
                )
                await cropped.process()
                
        except Exception as e:
            logger.error(f"Failed to process image {self.source_path}: {e}")
            raise