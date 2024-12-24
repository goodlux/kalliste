"""Handles cropped image processing for detected regions."""
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import asyncio
import subprocess
from PIL import Image
import math
import time
from .types import Detection, KallisteTag, ProcessingStatus, TagResult
from .model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

class CroppedImage:
    """Processes and manages cropped images from detections."""
    
    # Expansion factors for different detection types
    EXPANSION_FACTORS = {
        'face': 1.4,  # 40% expansion
        'person': 1.1,  # 10% expansion
        'default': 1.05  # 5% default expansion
    }
    
    def __init__(self, source_path: Path, output_dir: Path, detection: Detection, parent_tags: Optional[List[KallisteTag]] = None):
        """Initialize CroppedImage with optional parent tags."""
        logger.debug(f"Initializing CroppedImage for {detection.type} from {source_path}")
        self.source_path = source_path
        self.output_dir = output_dir
        self.detection = detection
        self.kalliste_tags = list(parent_tags or [])  # Create new list from parent tags
        self.output_path: Optional[Path] = None
        self.status = ProcessingStatus.PENDING
        self.cropped_image = None
        self.config_file = Path(__file__).parents[2] / "config" / "exiftool" / "kalliste.config"
        
    async def process(self):
        """Process this crop asynchronously."""
        logger.info(f"Starting crop processing for {self.detection.type} from {self.source_path}")
        self.status = ProcessingStatus.PROCESSING
        
        try:
            # First crop and save
            await self.crop_image()
            if not self.output_path:
                self.output_path = self._generate_output_path()
            await self.save()
            
            # Ensure file is fully written before proceeding
            await self._wait_for_file()
            
            # Run taggers after ensuring file exists
            await self.run_taggers()
            
            # Copy metadata from original and add our tags
            await self.copy_metadata_from_source()
            await self.inject_kalliste_tags()
            
            self.status = ProcessingStatus.COMPLETE
            logger.info(f"Completed processing crop from {self.source_path}")
            await self.on_complete()
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Error processing crop from {self.source_path}: {e}", exc_info=True)
            raise RuntimeError(f"Crop processing failed: {e}") from e

    async def _wait_for_file(self, max_retries: int = 5, delay: float = 0.1):
        """Wait for file to be fully written to disk."""
        loop = asyncio.get_event_loop()
        retries = 0
        
        while retries < max_retries:
            try:
                exists = await loop.run_in_executor(
                    None,
                    lambda: self.output_path.exists() and self.output_path.stat().st_size > 0
                )
                
                if exists:
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: Image.open(self.output_path).verify()
                        )
                        logger.debug(f"File {self.output_path} verified successfully")
                        return
                    except Exception as e:
                        logger.debug(f"File not yet ready (attempt {retries + 1}): {e}")
                
                retries += 1
                await asyncio.sleep(delay)
            except Exception as e:
                logger.debug(f"Error checking file (attempt {retries + 1}): {e}")
                retries += 1
                await asyncio.sleep(delay)
        
        raise RuntimeError(f"File {self.output_path} not ready after {max_retries} attempts")

    def expand_bbox(self, bbox: Tuple[int, int, int, int], 
                   image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Expand bounding box by detection type factor while maintaining aspect ratio."""
        x1, y1, x2, y2 = bbox
        im_width, im_height = image_size
        
        factor = self.EXPANSION_FACTORS.get(
            self.detection.type, 
            self.EXPANSION_FACTORS['default']
        )
        logger.debug(f"Using expansion factor {factor} for {self.detection.type}")
        
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_width = width * factor
        new_height = height * factor
        
        new_x1 = max(0, math.floor(center_x - new_width / 2))
        new_y1 = max(0, math.floor(center_y - new_height / 2))
        new_x2 = min(im_width, math.ceil(center_x + new_width / 2))
        new_y2 = min(im_height, math.ceil(center_y + new_height / 2))
        
        logger.debug(f"Expanded bbox from {bbox} to ({new_x1}, {new_y1}, {new_x2}, {new_y2})")
        return (new_x1, new_y1, new_x2, new_y2)
        
    async def crop_image(self):
        """Apply cropping based on detection."""
        logger.debug(f"Starting image crop for {self.detection.type}")
        loop = asyncio.get_event_loop()
        try:
            self.cropped_image = await loop.run_in_executor(
                None,
                self._crop_image_sync
            )
            logger.debug("Image cropping completed successfully")
        except Exception as e:
            logger.error(f"Error during image cropping: {e}", exc_info=True)
            raise RuntimeError(f"Image cropping failed: {e}") from e
        
    def _crop_image_sync(self) -> Image.Image:
        """Synchronous image cropping operations."""
        try:
            with Image.open(self.source_path) as img:
                expanded_bbox = self.expand_bbox(
                    self.detection.bbox,
                    img.size
                )
                
                cropped = img.crop(expanded_bbox)
                
                # Add crop-specific tags
                self.kalliste_tags.extend([
                    KallisteTag(
                        tag=f"crop_type:{self.detection.type}",
                        source="cropper",
                        confidence=self.detection.confidence
                    ),
                    KallisteTag(
                        tag=f"crop_expansion:{self.EXPANSION_FACTORS.get(self.detection.type, self.EXPANSION_FACTORS['default'])}",
                        source="cropper"
                    )
                ])
                
                return cropped
        except Exception as e:
            logger.error(f"Error in synchronous crop operation: {e}", exc_info=True)
            raise RuntimeError(f"Synchronous crop operation failed: {e}") from e
            
    def _generate_output_path(self) -> Path:
        """Generate output path preserving original format."""
        stem = self.source_path.stem
        ext = self.source_path.suffix
        return self.output_dir / f"{stem}_{self.detection.type}_crop{ext}"
            
    async def run_taggers(self):
        """Run appropriate taggers for this detection type."""
        logger.info(f"Running taggers for {self.detection.type} crop")
        
        try:
            tagger = ModelRegistry.get_tagger()
            logger.debug("Retrieved tagger from registry")
            
            tag_results = await tagger.tag_image(
                self.output_path,
                self.detection.type
            )
            logger.debug(f"Tagging complete with {len(tag_results)} category results")
            
            self._add_tag_results(tag_results)
            
        except Exception as e:
            logger.error(f"Error during tagging: {e}", exc_info=True)
            raise RuntimeError(f"Image tagging failed: {e}") from e
        
    def _add_tag_results(self, results: Dict[str, List[TagResult]]):
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

    async def copy_metadata_from_source(self):
        """Copy metadata from original image using exiftool."""
        if not self.output_path:
            raise ValueError("Output path not set")
            
        logger.debug(f"Copying metadata from {self.source_path} to {self.output_path}")
        cmd = [
            "exiftool",
            "-config", str(self.config_file),
            "-overwrite_original",  # Must be before file parameters
            "-TagsFromFile", str(self.source_path),
            "-all:all",
            "--FileSize",
            "--FileType",
            "--FileTypeExtension",
            "--ImageWidth",
            "--ImageHeight",
            "--ImageSize",
            "--PixelDimensions",
            str(self.output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"ExifTool error: {error_msg}")
                raise RuntimeError(f"ExifTool failed: {error_msg}")
                
            logger.debug("Metadata copy completed successfully")
            
        except Exception as e:
            logger.error(f"Error copying metadata: {e}", exc_info=True)
            raise RuntimeError(f"Metadata copy failed: {e}") from e
        
    async def inject_kalliste_tags(self):
        """Add Kalliste-specific tags."""
        if not self.kalliste_tags:
            logger.debug("No Kalliste tags to inject")
            return
            
        logger.debug(f"Injecting {len(self.kalliste_tags)} Kalliste tags")
        
        # Build the tag commands
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
            "-config", str(self.config_file),  # Config must be first
            "-overwrite_original",  # Next most important global flag
        ] + tag_commands + [str(self.output_path)]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"ExifTool error while injecting tags: {error_msg}")
                raise RuntimeError(f"Tag injection failed: {error_msg}")
                
            logger.debug("Kalliste tags injected successfully")
            
        except Exception as e:
            logger.error(f"Error injecting Kalliste tags: {e}", exc_info=True)
            raise RuntimeError(f"Tag injection failed: {e}") from e
        
    async def save(self):
        """Save final cropped image."""
        if not self.cropped_image:
            logger.error("No cropped image to save")
            raise ValueError("No cropped image to save")
            
        if not self.output_path:
            self.output_path = self._generate_output_path()
            logger.debug(f"Generated final output path: {self.output_path}")
            
        # Save in executor to not block
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._save_sync
            )
            logger.debug(f"Image saved successfully to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving image: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save image: {e}") from e
        
    def _save_sync(self):
        """Synchronous save operations."""
        if self.cropped_image.mode != 'RGB':
            self.cropped_image = self.cropped_image.convert('RGB')

        # We'll use PNG for better quality
        format = 'PNG'
        try:
            # Use optimize=True for better compression, but mainly to ensure proper file format
            self.cropped_image.save(
                self.output_path,
                format=format,
                optimize=True,  # This also helps prevent CRC errors
                icc_profile=self.cropped_image.info.get('icc_profile')  # Preserve color profile
            )
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            # If PNG fails, try JPEG as fallback
            fallback_path = self.output_path.with_suffix('.jpg')
            logger.warning(f"Attempting fallback save as JPEG to {fallback_path}")
            self.output_path = fallback_path
            self.cropped_image.save(
                self.output_path,
                format='JPEG',
                quality=95,
                optimize=True
            )
        
    async def on_complete(self):
        """Called when processing is complete."""
        logger.info(f"Cropped image {self.output_path} complete")

    
                     








