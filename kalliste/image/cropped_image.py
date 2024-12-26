"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image
import subprocess
import logging
from .types import KallisteTag, ProcessingStatus, TagResult
from ..detectors.base import Region

logger = logging.getLogger(__name__)

class CroppedImage:
    """Processes and manages cropped images from detections."""
    
    # Expansion factors for different detection types
    EXPANSION_FACTORS = {
        'face': 1.4,  # 40% expansion
        'person': 1.1,  # 10% expansion
        'default': 1.05  # 5% default expansion
    }
    
    def __init__(self, source_path: Path, output_dir: Path, region: Region):
        """Initialize CroppedImage."""
        self.source_path = source_path
        self.output_dir = output_dir
        self.region = region
        self.kalliste_tags = []
        self.output_path = self._generate_output_path()
        self.config_file = Path(__file__).parents[2] / "config" / "exiftool" / "kalliste.config"
        
    def _generate_output_path(self) -> Path:
        """Generate output path for the cropped image."""
        stem = self.source_path.stem
        ext = self.source_path.suffix
        return self.output_dir / f"{stem}_{self.region.region_type}_crop{ext}"
        
    async def process(self):
        """Process this crop."""
        logger.info(f"Starting crop processing for {self.region.region_type} from {self.source_path}")
        
        try:
            # Create and save the cropped image
            cropped_image = self._create_crop()
            self._save_crop(cropped_image)
            
            # Run taggers on the saved crop
            await self._run_taggers()
            
            # Copy metadata from original
            self._copy_metadata()
            
            # Add Kalliste tags
            self._add_kalliste_tags()
            
            logger.info(f"Completed processing crop from {self.source_path}")
            
        except Exception as e:
            logger.error(f"Error processing crop: {e}", exc_info=True)
            raise
            
    def _create_crop(self) -> Image.Image:
        """Create the cropped image."""
        with Image.open(self.source_path) as img:
            # Get expanded bbox
            x1, y1, x2, y2 = self.region.x1, self.region.y1, self.region.x2, self.region.y2
            factor = self.EXPANSION_FACTORS.get(self.region.region_type, self.EXPANSION_FACTORS['default'])
            
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            new_width = width * factor
            new_height = height * factor
            
            new_x1 = max(0, int(center_x - new_width / 2))
            new_y1 = max(0, int(center_y - new_height / 2))
            new_x2 = min(img.width, int(center_x + new_width / 2))
            new_y2 = min(img.height, int(center_y + new_height / 2))
            
            return img.crop((new_x1, new_y1, new_x2, new_y2))
            
    def _save_crop(self, image: Image.Image):
        """Save the cropped image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.save(
            self.output_path,
            format='PNG',
            optimize=True,
            icc_profile=image.info.get('icc_profile')
        )
        
    async def _run_taggers(self):
        """Run ML models to analyze the cropped image."""
        logger.info(f"Running taggers for {self.region.region_type} crop")
        
        try:
            # Get tagger from registry
            tagger = ModelRegistry.get_tagger()
            logger.debug("Retrieved tagger from registry")
            
            # Run the tagger - this is async because ML inference might be expensive
            tag_results = await tagger.tag_image(self.output_path, self.region.region_type)
            logger.debug(f"Tagging complete with {len(tag_results)} results")
            
            # Convert results to Kalliste tags
            self._convert_tag_results(tag_results)
            
        except Exception as e:
            logger.error(f"Error during tagging: {e}", exc_info=True)
            raise RuntimeError(f"Image tagging failed: {e}") from e
            
    def _convert_tag_results(self, results: Dict[str, List[TagResult]]):
        """Convert tagger results to Kalliste tags."""
        logger.debug(f"Converting {len(results)} tag results to Kalliste tags")
        for category, tags in results.items():
            for tag in tags:
                self.kalliste_tags.append(
                    KallisteTag(
                        tag=f"{category}:{tag.label}",
                        source=f"{category}_tagger",
                        confidence=tag.confidence
                    )
                )
        
    def _copy_metadata(self):
        """Copy metadata from original image."""
        cmd = [
            "exiftool",
            "-config", str(self.config_file),
            "-overwrite_original",
            "-TagsFromFile", str(self.source_path),
            "-all:all",
            "--FileSize",
            "--FileType",
            "--ImageWidth",
            "--ImageHeight",
            "--ImageSize",
            "--PixelDimensions",
            str(self.output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Metadata copy failed: {result.stderr}")
            
    def _add_kalliste_tags(self):
        """Add all Kalliste tags to the image."""
        # Add crop-specific tags first
        self.kalliste_tags.extend([
            KallisteTag(
                tag=f"crop_type:{self.region.region_type}",
                source="cropper",
                confidence=self.region.confidence
            ),
            KallisteTag(
                tag=f"crop_expansion:{self.EXPANSION_FACTORS.get(self.region.region_type, self.EXPANSION_FACTORS['default'])}",
                source="cropper"
            )
        ])
        
        # Add any tags from the region itself
        if hasattr(self.region, 'tags'):
            for tag_type, tag_value in self.region.tags:
                self.kalliste_tags.append(
                    KallisteTag(
                        tag=f"{tag_type}:{tag_value}",
                        source="exif_matching"
                    )
                )
        
        if not self.kalliste_tags:
            return
            
        # Build tag commands for all tags
        tag_commands = []
        for kt in self.kalliste_tags:
            tag_commands.extend([
                f"-XMP-Kalliste:Tag+={kt.tag}",
                f"-XMP-Kalliste:TagSource+={kt.source}"
            ])
            if kt.confidence is not None:
                tag_commands.append(
                    f"-XMP-Kalliste:TagConfidence+={kt.confidence}"
                )
                
        cmd = [
            "exiftool",
            "-config", str(self.config_file),
            "-overwrite_original"
        ] + tag_commands + [str(self.output_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Tag injection failed: {result.stderr}")
