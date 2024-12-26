"""DEPRECATED: Just leaving here in case there is anything useful to pull from this file """
"""Detection and tagging pipeline combining YOLO detection with image tagging."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from PIL import Image

from ..detectors.base import Region, DetectionConfig
from .taggers import ImageTagger, get_default_device
from .region_processor import RegionProcessor
from .crop_processor import CropProcessor
from ..config import YOLO_PERSON_MODEL, YOLO_FACE_MODEL

logger = logging.getLogger(__name__)

class DetectionPipeline_BAK:
    """Combines detection and tagging into a single pipeline."""
    
    def __init__(self, 
                 model: str = YOLO_PERSON_MODEL, # TODO: Remove, should be handled by detection_config
                 face_model: Optional[str] = YOLO_FACE_MODEL, # TODO: Remove, should be handled by detection_config
                 detection_config: List[DetectionConfig] = None,
                 device: Optional[str] = None): # TODO: Remove, should be handled by detection_config
        """
        Initialize the detection pipeline.
        TODO: We need a reference to the original saved file location here. 
        Args:
            model: YOLO model name (defaults to config.YOLO_PERSON_MODEL)
            face_model: Optional face detection model (defaults to config.YOLO_FACE_MODEL)
            detection_config: List of detection configurations
            device: Optional device specification ('mps', 'cuda', 'cpu', or None)
        """
        self.device = device or get_default_device()
        logger.info(f"Initializing pipeline on device: {self.device}")
        
        # Initialize processors
        self.crop_processor = CropProcessor(
            person_model_path=model,
            face_model_path=face_model
        )
        self.tagger = ImageTagger(device=self.device)
        logger.info("Detection pipeline initialized")
    
    def _prepare_kalliste_metadata(self, 
                                tags: Dict[str, Any], 
                                region: Region,
                                photoshoot_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepare Kalliste metadata from tags and region info."""
        # Extract orientation from tag results if available
        orientation_tag = None
        if 'orientation' in tags:
            orientation_results = tags['orientation']
            if orientation_results:
                # Get highest confidence orientation
                orientation_tag = max(orientation_results, 
                                   key=lambda x: x.confidence).label
        
        # Extract WD14 tags
        wd_tags = []
        if 'wd14' in tags:
            wd_tags = [tag.label for tag in tags['wd14']]
        
        metadata = {
            "photoshoot_id": photoshoot_id or "unknown",
            "caption": tags.get('caption', ''),
            "wd_tags": wd_tags,
            "orientation_tag": orientation_tag or 'unknown',
            "crop_type": region.region_type,
            "process_version": "1.0"
        }
        
        return metadata
    
    async def process_region(self, 
                           exported_image,
                           region: Region,
                           base_output_dir: Path,
                           photoshoot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Process a single detected region through the full pipeline."""
        # Create subdirectories for each stage
        full_res_dir = base_output_dir / "full_res"
        final_dir = base_output_dir / "sdxl"
        full_res_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save full-resolution crop
        full_res_paths = self.crop_processor.save_crops(
            image=exported_image,
            regions=[region],
            output_dir=full_res_dir,
            perform_resize=False,
            add_metadata=False
        )
        
        if not full_res_paths:
            logger.warning(f"Failed to create full-res crop for region {region}")
            return None
            
        full_res_path = full_res_paths[0]  # We only passed one region
        
        # 2. Run tagging on full-resolution crop
        tags = await self.tagger.tag_image(full_res_path)
        
        # 3. Prepare Kalliste metadata
        kalliste_metadata = self._prepare_kalliste_metadata(
            tags=tags,
            region=region,
            photoshoot_id=photoshoot_id
        )
        
        # 4. Create final SDXL-sized crop with metadata
        final_paths = self.crop_processor.save_crops(
            image=exported_image,
            regions=[region],
            output_dir=final_dir,
            perform_resize=True,    # Resize to SDXL dimensions
            add_metadata=True,      # Add Kalliste metadata
            kalliste_metadata=kalliste_metadata
        )
        
        if not final_paths:
            logger.warning(f"Failed to create final SDXL crop for region {region}")
            return None
            
        final_path = final_paths[0]  # We only passed one region
        
        # Return all the information
        return {
            'region': region.__dict__,
            'tags': tags,
            'full_res_path': str(full_res_path),
            'final_path': str(final_path),
            'kalliste_metadata': kalliste_metadata
        }
    

    # TODO: This is the entry point for the detection pipeline. This could be renamed "create" or just make it clear that this is the main function.
    async def process_image(self, 
                          image_path: Path,
                          output_dir: Path,
                          photoshoot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a single image through the detection pipeline.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save processed crops
            photoshoot_id: Optional photoshoot identifier for metadata
            
        Returns:
            List of dictionaries containing detection regions, tags, and paths
        """
        # Ensure paths are Path objects
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ExportedImage object - temporary fix until we update exported_image
        class ExportedImage:
            def __init__(self, source_path):
                self.source_path = source_path
        
        # Create ExportedImage object
        exported_image = ExportedImage(source_path=image_path)
        
        # Run detection using CropProcessor
        regions = self.crop_processor.process_image(exported_image)
        
        # Process each region
        tasks = [
            self.process_region(exported_image, region, output_dir, photoshoot_id)
            for region in regions
        ]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        return [r for r in results if r is not None]