"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict
import uuid
import logging
from PIL import Image

from ..region import Region, RegionExpander, RegionDownsizer
from ..taggers.tagger_pipeline import TaggerPipeline
from ..types import TagResult
from .caption_file_writer import CaptionFileWriter
from .exif_writer import ExifWriter
from ..tag.kalliste_tag import KallisteStringTag

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
        
    async def process(self) -> Optional[Dict]:
        """
        Process the cropped region.
        Returns:
            Optional[Dict]: Processing statistics including all assessments and rejection info
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            stats = {
                'rejected_small': False,
                'technical': None,
                'aesthetic': None,
                'overall': None,
                'kalliste': None
            }
            
            with Image.open(self.source_path) as img:
                logger.info(f"Opened image {self.source_path}")
                
                logger.info("Expanding region to SDXL ratios")
                expanded_region = RegionExpander.expand_region_to_sdxl_ratios(
                    self.region,
                    img.width,
                    img.height
                )
                
                logger.info("Validating size")
                if not RegionDownsizer.is_valid_size(expanded_region, img):
                    logger.info(f"Region too small for SDXL, skipping: {expanded_region}")
                    stats['rejected_small'] = True
                    return stats
                
                logger.info("Cropping image")
                cropped = img.crop((
                    expanded_region.x1,
                    expanded_region.y1,
                    expanded_region.x2,
                    expanded_region.y2
                ))
                
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
                
                # Get quality assessments for statistics
                tech_tag = self.region.kalliste_tags.get("KallisteNimaTechnicalAssessment")
                if tech_tag:
                    stats['technical'] = tech_tag.value
                    
                aes_tag = self.region.kalliste_tags.get("KallisteNimaAestheticAssessment")
                if aes_tag:
                    stats['aesthetic'] = aes_tag.value
                    
                overall_tag = self.region.kalliste_tags.get("KallisteNimaOverallAssessment")
                if overall_tag:
                    stats['overall'] = overall_tag.value
                
                logger.info("Resizing to SDXL")
                sdxl_image = RegionDownsizer.downsize_to_sdxl(cropped)

                # Determine Kalliste assessment
                assessment = self._determine_kalliste_assessment(self.region)
                stats['kalliste'] = assessment
                
                # Add the assessment to tags
                self.region.add_tag(KallisteStringTag(
                    "KallisteAssessment",
                    assessment
                ))
                
                # Save the image to the accept/reject assessment folder
                logger.info(f"Saving image to {assessment} folder")
                output_filename = f"{self.source_path.stem}_{self.region.region_type}_{uuid.uuid4()}.png"
                quality_dir = self.output_dir / assessment
                quality_dir.mkdir(exist_ok=True)
                output_path = quality_dir / output_filename
                sdxl_image.save(output_path, "PNG", optimize=True)

                # Copy the metadata from original image, add kalliste tags, write the caption file
                logger.info("Writing metadata")
                self._write_metadata(output_path)
                logger.info("Metadata written")
                
                return stats
                
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