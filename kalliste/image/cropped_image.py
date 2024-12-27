"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict

import subprocess
import logging
from ..types import KallisteTag, ProcessingStatus, TagResult
from ..detectors.base import Region

from ..processors.metadata_processor import MetadataProcessor
import uuid
# kalliste/image/cropped_image.py

from PIL import Image
import uuid
import logging
from ..processors.image_resizer import SDXLResizer
from ..processors.metadata_processor import MetadataProcessor
from ..types import TagResult
from ..taggers.tagger_pipeline import TaggerPipeline

logger = logging.getLogger(__name__)

class CroppedImage:
    """Processes and manages cropped images from detections."""
    
    # Expansion factors for different detection types
    EXPANSION_FACTORS = {
        'face': 1.4,  # 40% expansion
        'person': 1.1,  # 10% expansion
        'default': 1.05  # 5% default expansion
    }
    
    def __init__(self, 
                source_path: Path, 
                output_dir: Path, 
                region: Region, 
                config: Dict,
                tagger_pipeline: Optional[TaggerPipeline] = None):
        """Initialize CroppedImage.
        
        Args:
            source_path: Path to original image
            output_dir: Directory to save cropped image
            region: Region to crop
            config: Configuration dictionary
            tagger_pipeline: Optional TaggerPipeline instance (can be shared)
        """
        self.source_path = source_path
        self.output_dir = output_dir
        self.region = region
        self.config = config
        self.kalliste_tags = []
        
        # Initialize processors
        self.resizer = SDXLResizer()
        self.metadata_processor = MetadataProcessor()
        self.tagger_pipeline = tagger_pipeline or TaggerPipeline(config)
        
    async def process(self):
        """Process the cropped region.
        
        Flow:
        1. Crop region (already properly padded and ratio'd)
        2. Resize to SDXL dimensions and save
        3. Run taggers on saved image
        4. Copy metadata and add Kalliste tags
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Crop, resize, and save
            final_path = self._crop_resize_and_save()
            
            # Run taggers on the saved image
            await self._run_taggers(final_path)
            
            # Copy metadata from original and add Kalliste tags
            self._copy_metadata(final_path)
            
        except Exception as e:
            logger.error(f"Failed to process cropped image: {e}", exc_info=True)
            raise

    def _crop_resize_and_save(self) -> Path:
        """Crop region, resize to SDXL dimensions, and save."""
        with Image.open(self.source_path) as img:
            # Crop
            cropped = img.crop((
                self.region.x1, self.region.y1,
                self.region.x2, self.region.y2
            ))
            
            # Generate output filename
            filename = f"{self.region.region_type}_{uuid.uuid4()}.png"
            final_path = self.output_dir / filename
            
            # Resize and save using SDXLResizer
            self.resizer.resize_image(cropped, final_path)
            
            return final_path

    def _copy_metadata(self, final_path: Path):
        """Copy metadata from original and add Kalliste-specific tags."""
        # Prepare Kalliste metadata
        kalliste_metadata = {
            'region_type': self.region.region_type,
            'confidence': self.region.confidence,
            'original_path': str(self.source_path),
            'process_version': '1.0',  # or from config
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
        
        # Copy metadata and add Kalliste tags
        success = self.metadata_processor.copy_metadata(
            source_path=self.source_path,
            dest_path=final_path,
            kalliste_metadata=kalliste_metadata
        )
        
        if not success:
            logger.warning(f"Failed to copy metadata for {final_path}")
