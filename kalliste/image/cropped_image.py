"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict, Set
import uuid
import logging
from PIL import Image

from ..region import Region, RegionExpander, RegionDownsizer
from ..taggers.tagger_pipeline import TaggerPipeline
from ..types import TagResult
from .caption_file_writer import CaptionFileWriter
from .exif_writer import ExifWriter
from ..tag.kalliste_tag import  KallisteStringTag, KallisteBagTag

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
        
    async def process(self):
        """Process the cropped region."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
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
                    return None
                
                logger.info("Cropping image")
                cropped = img.crop((
                    expanded_region.x1,
                    expanded_region.y1,
                    expanded_region.x2,
                    expanded_region.y2
                ))
                
                logger.info("Running taggers")
                tag_results = await self.tagger_pipeline.tag_pillow_image(
                    image=cropped,
                    region_type=self.region.region_type
                )
                
                # Log raw tagger results for verification
                logger.info("Raw tagger results:")
                for key, value in tag_results.items():
                    logger.info(f"  {key}: {value}")
                
                self._process_tag_results(tag_results)
                
                # Log processed kalliste_tags
                logger.info("Processed kalliste_tags:")
                for tag_name, tag_values in self.region.kalliste_tags.items():
                    logger.info(f"  {tag_name}: {tag_values}")
                
                logger.info("Resizing to SDXL")
                sdxl_image = RegionDownsizer.downsize_to_sdxl(cropped)
                
                logger.info("Saving image")
                output_filename = f"{self.source_path.stem}_{self.region.region_type}_{uuid.uuid4()}.png"
                output_path = self.output_dir / output_filename
                sdxl_image.save(output_path, "PNG", optimize=True)
                
                logger.info("Writing metadata")
                self._write_metadata(output_path)
                logger.info("Metadata written")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to process cropped image: {e}")
            raise
            
    def _process_tag_results(self, tag_results: Dict[str, List[TagResult]]) -> None:
        """Process tagger results into region kalliste_tags."""
        try:
            logger.info("Processing tag results:")
            
            # Process orientation tags
            if 'orientation' in tag_results:
                orientations = tag_results['orientation']
                if orientations:
                    logger.info("Processing orientation tags:")
                    logger.info(f"  All orientations: {orientations}")
                    # Get highest confidence orientation
                    highest_conf = max(orientations, key=lambda x: x.confidence)
                    orientation_tag = KallisteStringTag(
                        "KallisteOrientationTag",
                        highest_conf.label.lower()
                    )
                    self.region.add_tag(orientation_tag)
                    
                    # Save raw orientation data as bag
                    raw_orientations = {
                        f"{tag.label}({tag.confidence:.2f})" 
                        for tag in orientations
                    }
                    raw_tag = KallisteBagTag(
                        "KallisteRawOrientationData",
                        raw_orientations
                    )
                    self.region.add_tag(raw_tag)
            
            # Process wd14 tags (category 0)
            if '0' in tag_results:
                logger.info("Processing wd14 tags:")
                # Get tags without confidences for SDXL
                wd14_tags = {tag.label for tag in tag_results['0']}
                logger.info(f"  Raw wd14 tags: {tag_results['0']}")
                logger.info(f"  Processed wd14 tags: {wd14_tags}")
                
                tag = KallisteBagTag("KallisteWd14Tags", wd14_tags)
                self.region.add_tag(tag)
                
                # Save raw WD14 data with confidences as bag
                raw_wd14 = {
                    f"{tag.label}({tag.confidence:.2f})" 
                    for tag in tag_results['0']
                }
                raw_tag = KallisteBagTag("KallisteRawWd14Tags", raw_wd14)
                self.region.add_tag(raw_tag)
            
            # Process caption
            if 'caption' in tag_results and tag_results['caption']:
                logger.info("Processing caption:")
                caption = tag_results['caption'][0].label
                logger.info(f"  Caption: {caption}")
                caption_tag = KallisteStringTag("KallisteCaption", caption)
                self.region.add_tag(caption_tag)

            # Add original path
            path_tag = KallisteStringTag(
                "KallisteOriginalPath", 
                str(self.source_path)
            )
            self.region.add_tag(path_tag)

        except Exception as e:
            logger.error(f"Error processing tag results: {e}")
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