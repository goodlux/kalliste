"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict
import uuid
import logging
from PIL import Image

from ..region import Region, RegionExpander, RegionDownsizer
from ..processors.metadata_processor import MetadataProcessor
from ..taggers.tagger_pipeline import TaggerPipeline
from ..types import TagResult

logger = logging.getLogger(__name__)

class CroppedImage:
    """Processes and manages cropped images from detections."""
    
    def __init__(self, 
                source_path: Path, 
                output_dir: Path, 
                region: Region, 
                config: Dict,
                tagger_pipeline: Optional[TaggerPipeline] = None):
        """Initialize CroppedImage."""
        self.source_path = source_path
        self.output_dir = output_dir
        self.region = region
        self.config = config
        self.kalliste_tags = []
        
        # Initialize processors
        self.metadata_processor = MetadataProcessor()
        self.tagger_pipeline = tagger_pipeline or TaggerPipeline(config)
        
    async def process(self):
        """Process the cropped region.
        
        Flow:
        1. Expand region to fit SDXL ratios
        2. Validate size
        3. Crop image
        4. Run taggers on cropped PIL Image
        5. Resize to SDXL
        6. Save
        7. Add metadata with tags
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with Image.open(self.source_path) as img:
                logger.info(f"Opened image {self.source_path}")
                
                # 1. Expand region to fit SDXL ratios
                logger.info("Expanding region to SDXL ratios")
                expanded_region = RegionExpander.expand_region_to_sdxl_ratios(
                    self.region,
                    img.width,
                    img.height
                )
                logger.info(f"Expanded region: {expanded_region}")
                
                # 2. Validate size
                logger.info("Validating size")
                if not RegionDownsizer.is_valid_size(expanded_region, img):
                    logger.info(f"Region too small for SDXL, skipping: {expanded_region}")
                    return None
                
                # 3. Crop image
                logger.info("Cropping image")
                cropped = img.crop((
                    expanded_region.x1,
                    expanded_region.y1,
                    expanded_region.x2,
                    expanded_region.y2
                ))
                logger.info(f"Cropped image size: {cropped.size}")
                
                # 4. Run taggers on PIL Image directly
                logger.info("Running taggers")
                tag_results = await self.tagger_pipeline.tag_pillow_image(
                    image=cropped,
                    region_type=self.region.region_type
                )
                logger.info(f"Got tag results: {tag_results}")
                self.kalliste_tags = self._process_tag_results(tag_results)
                
                # 5. Resize to SDXL
                logger.info("Resizing to SDXL")
                sdxl_image = RegionDownsizer.downsize_to_sdxl(cropped)
                logger.info(f"Resized image size: {sdxl_image.size}")
                
                # 6. Save
                logger.info("Saving image")
                output_filename = f"{self.source_path.stem}_{self.region.region_type}_{uuid.uuid4()}.png"
                output_path = self.output_dir / output_filename
                sdxl_image.save(output_path, "PNG", optimize=True)
                logger.info(f"Saved to {output_path}")
                
                # 7. Add metadata
                logger.info("Adding metadata")
                self._add_metadata(output_path)
                logger.info("Metadata added")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to process cropped image: {e}")
            raise
            
    def _process_tag_results(self, tag_results: Dict[str, List[TagResult]]) -> List[Dict]:
        """Convert tagger results to Kalliste tag format."""
        kalliste_tags = []
        for tagger_name, results in tag_results.items():
            for tag in results:
                if isinstance(tag, dict):
                    kalliste_tags.append({
                        'label': tag['label'],
                        'confidence': tag.get('confidence'),
                        'category': tag.get('category'),
                        'tagger': tagger_name
                    })
        return kalliste_tags
        
    def _add_metadata(self, image_path: Path):
        """Add Kalliste metadata including tags to the image."""
        kalliste_metadata = {
            'region_type': self.region.region_type,
            'confidence': self.region.confidence,
            'original_path': str(self.source_path),
            'process_version': '1.0',
            'tags': [tag['label'] for tag in self.kalliste_tags],
            'tag_confidences': {
                tag['label']: tag['confidence'] 
                for tag in self.kalliste_tags 
                if tag.get('confidence')
            },
            'tag_categories': {
                tag['label']: tag['category'] 
                for tag in self.kalliste_tags 
                if tag.get('category')
            },
            'tag_sources': {
                tag['label']: tag['tagger'] 
                for tag in self.kalliste_tags 
                if tag.get('tagger')
            }
        }
        
        success = self.metadata_processor.copy_metadata(
            source_path=self.source_path,
            dest_path=image_path,
            kalliste_metadata=kalliste_metadata
        )