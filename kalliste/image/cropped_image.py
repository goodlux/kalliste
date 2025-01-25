"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict
import uuid
import logging
from PIL import Image
import numpy as np

from ..region import Region, RegionExpander, RegionDownsizer
from ..taggers.tagger_pipeline import TaggerPipeline
from ..types import TagResult
from .caption_file_writer import CaptionFileWriter
from .exif_writer import ExifWriter
from ..tag.kalliste_tag import KallisteStringTag
import io
import base64

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
        
        # Initialize tagger
        self.tagger_pipeline = tagger_pipeline or TaggerPipeline(config)
        
    async def process(self) -> Dict:
        """Process the cropped region."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with Image.open(self.source_path) as img:
                logger.info(f"Opened image {self.source_path}")
                
                # Get original region dimensions for logging
                orig_width = self.region.x2 - self.region.x1
                orig_height = self.region.y2 - self.region.y1
                logger.info(f"Original region dimensions: {orig_width}x{orig_height}")
                
                logger.info("Expanding region to SDXL ratios")
                expanded_region = RegionExpander.expand_region_to_sdxl_ratios(
                    self.region,
                    img.width,
                    img.height
                )
                
                # Get expanded dimensions for logging
                exp_width = expanded_region.x2 - expanded_region.x1
                exp_height = expanded_region.y2 - expanded_region.y1
                logger.info(f"Expanded region dimensions: {exp_width}x{exp_height}")
                
                # Validate size before proceeding
                if not RegionDownsizer.is_valid_size(expanded_region, img):
                    logger.info(f"Region too small for SDXL after expansion (dimensions: {exp_width}x{exp_height}), skipping")
                    return {
                        'rejected_small': True,
                        'technical': None,
                        'aesthetic': None,
                        'overall': None,
                        'kalliste': None
                    }
                
                logger.info("Cropping image")
                cropped = img.crop((
                    expanded_region.x1,
                    expanded_region.y1,
                    expanded_region.x2,
                    expanded_region.y2
                ))
                
                logger.info(f"Cropped dimensions: {cropped.width}x{cropped.height}")
                
                logger.info("Running taggers")
                await self.tagger_pipeline.tag_pillow_image(
                    image=cropped,
                    region_type=self.region.region_type,
                    region=self.region
                )
                
                # Add original path
                path_tag = KallisteStringTag(
                    "KallisteOriginalPath", 
                    str(self.source_path)
                )
                self.region.add_tag(path_tag)
                
                # Log all tags after processing
                logger.info("Final kalliste_tags:")
                for tag_name, tag_values in self.region.kalliste_tags.items():
                    logger.info(f"  {tag_name}: {tag_values}")
                
                logger.info("Resizing to SDXL")
                sdxl_image = RegionDownsizer.downsize_to_sdxl(cropped)
                logger.info(f"Final SDXL dimensions: {sdxl_image.width}x{sdxl_image.height}")

                # Determine Kalliste assessment and add the tag
                assessment = self._determine_kalliste_assessment(self.region)
                self.region.add_tag(KallisteStringTag(
                    "KallisteAssessment",
                    assessment
                ))
                
                # Save the image to the output directory
                logger.info(f"Saving image to output directory")
                output_filename = f"{self.source_path.stem}_{self.region.region_type}_{uuid.uuid4()}.png"
                output_path = self.output_dir / output_filename
                sdxl_image.save(output_path, "PNG", optimize=True)

                # Copy the metadata from original image, add kalliste tags, write the caption file
                logger.info("Writing metadata")
                self._write_metadata(output_path)
                logger.info("Metadata written")
                
                return {
                    'rejected_small': False,
                    'technical': self.region.get_tag_value('KallisteNimaTechnicalAssessment'),
                    'aesthetic': self.region.get_tag_value('KallisteNimaAestheticAssessment'),
                    'overall': self.region.get_tag_value('KallisteNimaOverallAssessment'),
                    'kalliste': assessment
                }
                
        except Exception as e:
            logger.error(f"Failed to process cropped image: {e}")
            raise
        
    def _write_metadata(self, image_path: Path):
        """Write region's kalliste_tags to both caption file and XMP metadata."""
        try:
            # Write caption file
            txt_path = image_path.with_suffix('.txt')
            caption_writer = CaptionFileWriter(txt_path)
            if not caption_writer.write_caption(self.region.kalliste_tags):
                logger.error("Failed to write caption file")
                
            # Write XMP metadata
            exif_writer = ExifWriter(self.source_path, image_path)
            if not exif_writer.write_tags(self.region.kalliste_tags):
                logger.error("Failed to write XMP metadata")
                
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")
            raise

    def _determine_kalliste_assessment(self, region: Region) -> str:
        """
        Determine if an image should be accepted or rejected based on NIMA assessments.
        Accepts if either:
        1. NIMA overall assessment is "acceptable" OR
        2. Technical quality is "high_quality" (overrides overall assessment)
        
        Returns:
            str: "accept" or "reject"
        """
        try:
            # Get technical quality assessment
            tech_assessment = region.kalliste_tags.get("KallisteNimaTechnicalAssessment")
            if tech_assessment and tech_assessment.value == "high_quality":
                logger.info("Accepting image due to high technical quality")
                return "accept"
            
            # Otherwise check overall assessment
            nima_overall = region.kalliste_tags.get("KallisteNimaOverallAssessment")
            if nima_overall and nima_overall.value == "acceptable":
                logger.info("Accepting image due to acceptable overall assessment")
                return "accept"
            
            # If neither condition is met, reject
            logger.info("Rejecting image: neither technically excellent nor acceptable overall")
            return "reject"
                    
        except Exception as e:
            logger.warning(f"Error determining Kalliste assessment: {e}. Defaulting to reject.")
            return "reject"