"""Handles processing of original images and creation of derived crops."""
from pathlib import Path
from typing import List, Optional, Dict
from .types import  ProcessingStatus
from .cropped_image import CroppedImage
from ..detectors.base import DetectionConfig, Region
from ..detectors.detection_pipeline import DetectionPipeline, DetectionResult
from .model_registry import ModelRegistry
import asyncio
import logging

logger = logging.getLogger(__name__)

class OriginalImage:
    def __init__(self, input_path: Path, output_path: Path, 
                 detection_config: Optional[Dict] = None,
                 model_registry=None):
        self.input_path = input_path
        self.output_path = output_path
        self.detection_config = detection_config or {}  # Empty dict if None
        self.model_registry = model_registry
        self.exif_metadata = {}  
        self.cropped_images: List[CroppedImage] = []
        self.status = ProcessingStatus.PENDING
        self.detected_regions: Optional[DetectionResult] = None

    async def process(self):
        """Main processing pipeline"""
        self.status = ProcessingStatus.PROCESSING
        logger.info(f"Starting processing for {self.input_path}")

        try:
            # Step 1: Load EXIF metadata
            await self._load_exif_metadata()
            
            # Step 2: Run detection pipeline
            try:
                await self._run_detection_pipeline()
                if self.detected_regions:
                    logger.info(f"Found {len(self.detected_regions.regions)} regions in {self.input_path}")
                else:
                    logger.info(f"No regions detected in {self.input_path}")
            except Exception as e:
                logger.error(f"Detection failed for {self.input_path}: {e}", exc_info=True)
                raise

           # Step 3: Match regions with metadata and assign tags
            await self._match_regions_with_exif()   
            
            # Step 4: Create and process crops for each region
            if self.detected_regions and self.detected_regions.regions:
                crop_tasks = []
                for region in self.detected_regions.regions:
                    cropped = CroppedImage(
                        source_path=self.input_path,
                        output_dir=self.output_path,
                        region=region  # Pass region directly
                    )
                    self.cropped_images.append(cropped)
                    crop_tasks.append(cropped.process())
                            
                # Wait for all crops to complete
                try:
                    await asyncio.gather(*crop_tasks)
                except Exception as e:
                    logger.error(f"Error processing crops for {self.input_path}: {e}", exc_info=True)
                    raise
                    
            self.status = ProcessingStatus.COMPLETE
            logger.info(f"Completed processing for {self.input_path}")
            # Signal completion
            await self.on_complete()
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error(f"Error processing {self.input_path}: {e}", exc_info=True)
            raise

    async def _run_detection_pipeline(self) -> None:
        """Run detection pipeline on image asynchronously"""
        try:
            # Create detection pipeline
            pipeline = DetectionPipeline(model_registry=self.model_registry)
            
            # Run pipeline in executor to not block
            loop = asyncio.get_event_loop()
            self.detected_regions = await loop.run_in_executor(
                None, 
                pipeline.detect,
                self.input_path,
                self.detection_config
            )
            
            logger.debug(
                f"Detection pipeline complete for {self.input_path}: "
                f"found {len(self.detected_regions.regions) if self.detected_regions else 0} regions"
            )
            
        except Exception as e:
            logger.error(f"Detection pipeline failed for {self.input_path}: {e}", exc_info=True)
            raise
        
    async def _load_exif_metadata(self) -> None:
        """Load selected EXIF metadata tags using exiftool."""
        try:
            # List of tags we want to extract
            requested_tags = [
                "Region Name", 
                "Region Type", 
                "Region Area H",
                "Region Area W",
                "Region Area X",
                "Region Area Y"  
            ]

            # Build exiftool command
            tag_args = [f"-{tag}" for tag in requested_tags]
            cmd = ["exiftool", "-j", "-n"] + tag_args + [str(self.input_path)]
            
            # Run exiftool in executor to not block
            loop = asyncio.get_event_loop()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                import json
                # ExifTool returns a JSON array, but we only have one image
                metadata = json.loads(stdout)[0]
                self.exif_metadata = metadata
                logger.debug(f"Loaded {len(metadata)} EXIF tags from {self.input_path}")
            else:
                error = stderr.decode().strip()
                logger.error(f"ExifTool failed: {error}")
                self.exif_metadata = {}
                
        except Exception as e:
            logger.error(f"Failed to load EXIF metadata: {e}")
            self.exif_metadata = {}        

    async def _match_regions_with_exif(self) -> None:
        """Match detected regions with EXIF region data."""
        if not self.detected_regions or not self.exif_metadata:
            return
            
        # Extract region data from EXIF metadata
        try:
            region_names = self.exif_metadata.get('Region Name', '').split(', ')
            region_types = self.exif_metadata.get('Region Type', '').split(', ')
            region_h = [float(h) for h in self.exif_metadata.get('Region Area H', '').split(', ')]
            region_w = [float(w) for w in self.exif_metadata.get('Region Area W', '').split(', ')]
            region_x = [float(x) for x in self.exif_metadata.get('Region Area X', '').split(', ')]
            region_y = [float(y) for y in self.exif_metadata.get('Region Area Y', '').split(', ')]
            
            # Check if we have valid region data
            if not all([region_names, region_types, region_h, region_w, region_x, region_y]):
                return
                
            # For each detected region
            for region in self.detected_regions.regions:
                # Convert EXIF coordinates to absolute pixel space to match Region format
                img_w = region.image_width
                img_h = region.image_height
                
                best_iou = 0
                best_match_name = None
                
                # Compare with each EXIF region
                for i in range(len(region_names)):
                    # Create Region object from EXIF data
                    exif_region = Region(
                        x1=region_x[i] * img_w,
                        y1=region_y[i] * img_h,
                        x2=(region_x[i] + region_w[i]) * img_w,
                        y2=(region_y[i] + region_h[i]) * img_h,
                        region_type=region_types[i].lower(),
                        confidence=1.0  # EXIF regions don't have confidence scores
                    )
                    
                    from ..image.utils import calculate_iou
                    iou = calculate_iou(region, exif_region)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_name = region_names[i]
                
                # If we found a good match (using 0.5 IoU threshold)
                if best_iou >= 0.5 and best_match_name:
                    region.tags.append(('person', best_match_name))
                    
        except Exception as e:
            logger.error(f"Error matching regions with EXIF data: {e}")

    async def on_complete(self):
        """Called when processing is complete"""
        logger.info(f"Original image {self.input_path} processing complete")
        
    async def on_crop_complete(self, crop_path: Path):
        """Called when a cropped image signals completion"""
        logger.info(f"Crop {crop_path} from {self.input_path} complete")