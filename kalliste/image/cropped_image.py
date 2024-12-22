from pathlib import Path
from typing import List, Optional, Tuple, Dict
import asyncio
import subprocess
from PIL import Image
import math
from .types import Detection, KallisteTag, ProcessingStatus
from ..taggers.tagger_pipeline import TaggerPipeline, TagResult

class CroppedImage:
    # Expansion factors for different detection types
    EXPANSION_FACTORS = {
        'face': 1.4,  # 40% expansion
        'person': 1.1,  # 10% expansion
        'default': 1.05  # 5% default expansion
    }
    
    def __init__(self, source_path: Path, detection: Detection):
        self.source_path = source_path
        self.detection = detection
        self.kalliste_tags: List[KallisteTag] = []
        self.output_path: Optional[Path] = None
        self.status = ProcessingStatus.PENDING
        self.cropped_image = None  # Will hold the PIL Image after cropping
        
        # Initialize tagger pipeline
        self.tagger_pipeline = TaggerPipeline()
        
    async def process(self):
        """Process this crop asynchronously"""
        self.status = ProcessingStatus.PROCESSING
        
        try:
            # 1. Apply cropping based on detection
            await self.crop_image()
            
            # 2. Run appropriate taggers
            await self.run_taggers()
            
            # 3. Copy metadata from original
            await self.copy_metadata_from_source()
            
            # 4. Add Kalliste tags
            await self.inject_kalliste_tags()
            
            # 5. Save final image
            await self.save()
            
            self.status = ProcessingStatus.COMPLETE
            # Signal completion
            await self.on_complete()
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            print(f"Error processing crop from {self.source_path}: {e}")
            raise

    def expand_bbox(self, bbox: Tuple[int, int, int, int], 
                   image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Expand bounding box by detection type factor while maintaining aspect ratio"""
        x1, y1, x2, y2 = bbox
        im_width, im_height = image_size
        
        # Get expansion factor based on detection type
        factor = self.EXPANSION_FACTORS.get(
            self.detection.type, 
            self.EXPANSION_FACTORS['default']
        )
        
        # Calculate current dimensions
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate new dimensions
        new_width = width * factor
        new_height = height * factor
        
        # Calculate new coordinates
        new_x1 = max(0, math.floor(center_x - new_width / 2))
        new_y1 = max(0, math.floor(center_y - new_height / 2))
        new_x2 = min(im_width, math.ceil(center_x + new_width / 2))
        new_y2 = min(im_height, math.ceil(center_y + new_height / 2))
        
        return (new_x1, new_y1, new_x2, new_y2)
        
    async def crop_image(self):
        """Apply cropping based on detection"""
        # Run image operations in executor to not block
        loop = asyncio.get_event_loop()
        self.cropped_image = await loop.run_in_executor(
            None,
            self._crop_image_sync
        )
        
    def _crop_image_sync(self) -> Image.Image:
        """Synchronous image cropping operations"""
        with Image.open(self.source_path) as img:
            # Expand bbox based on detection type
            expanded_bbox = self.expand_bbox(
                self.detection.bbox,
                img.size
            )
            
            # Crop image
            cropped = img.crop(expanded_bbox)
            
            # Add cropping info to Kalliste tags
            self.kalliste_tags.append(
                KallisteTag(
                    tag=f"crop_type:{self.detection.type}",
                    source="cropper",
                    confidence=self.detection.confidence
                )
            )
            
            # Add expansion factor to tags
            self.kalliste_tags.append(
                KallisteTag(
                    tag=f"crop_expansion:{self.EXPANSION_FACTORS.get(self.detection.type, self.EXPANSION_FACTORS['default'])}",
                    source="cropper"
                )
            )
            
            return cropped
            
    async def run_taggers(self):
        """Run appropriate taggers for this detection type"""
        # Save temporary image for taggers if needed
        if not self.output_path:
            # Generate output path if not set
            stem = self.source_path.stem
            self.output_path = self.source_path.parent / f"{stem}_{self.detection.type}_crop.jpg"
            self.cropped_image.save(self.output_path, "JPEG", quality=95)
            
        # Run taggers appropriate for this detection type
        tag_results = await self.tagger_pipeline.tag_image(
            self.output_path,
            self.detection.type
        )
        
        # Convert tag results to Kalliste tags
        self._add_tag_results(tag_results)
        
    def _add_tag_results(self, results: Dict[str, List[TagResult]]):
        """Convert tagger results to Kalliste tags"""
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
        """Copy metadata from original image using exiftool"""
        if not self.output_path:
            raise ValueError("Output path not set")
            
        cmd = [
            "exiftool",
            "-TagsFromFile", str(self.source_path),
            "-all:all",
            "-ImageSize=",
            "-PixelDimensions=",
            str(self.output_path)
        ]
        # Run exiftool asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
    async def inject_kalliste_tags(self):
        """Add Kalliste-specific tags"""
        if not self.kalliste_tags:
            return
            
        # Format tags for exiftool
        tag_commands = []
        for kt in self.kalliste_tags:
            # Create XMP:Kalliste:Tags array entry
            tag_commands.extend([
                f"-XMP:Kalliste:Tag+={kt.tag}",
                f"-XMP:Kalliste:TagSource+={kt.source}"
            ])
            if kt.confidence is not None:
                tag_commands.append(f"-XMP:Kalliste:TagConfidence+={kt.confidence}")
                
        cmd = ["exiftool"] + tag_commands + [str(self.output_path)]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
    async def save(self):
        """Save final cropped image"""
        if not self.cropped_image:
            raise ValueError("No cropped image to save")
            
        # Generate output path if not set
        if not self.output_path:
            stem = self.source_path.stem
            self.output_path = self.source_path.parent / f"{stem}_{self.detection.type}_crop.jpg"
            
        # Save in executor to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._save_sync
        )
        
    def _save_sync(self):
        """Synchronous save operations"""
        self.cropped_image.save(self.output_path, "JPEG", quality=95)
        
    async def on_complete(self):
        """Called when processing is complete"""
        print(f"Cropped image {self.output_path} complete")