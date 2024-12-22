"""Handles detection and cropping of persons and faces in images."""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import piexif
from PIL import Image
from ultralytics import YOLO
import torch
import logging
import requests

from kalliste.models.exported_image import ExportedImage, Region
from kalliste.processors.region_processor import RegionProcessor
from kalliste.processors.image_resizer import ImageResizer
from kalliste.processors.metadata_processor import MetadataProcessor
from kalliste.config import YOLO_PERSON_MODEL, YOLO_FACE_MODEL, YOLO_CACHE_DIR

logger = logging.getLogger(__name__)

class CropProcessor:
    def __init__(self, 
                 person_model_path: str = YOLO_PERSON_MODEL,
                 face_model_path: str = YOLO_FACE_MODEL,
                 confidence_threshold: float = 0.4,
                 iou_threshold: float = 0.7):
        
        """Initialize the CropProcessor with both person and face YOLO models."""
        logger.info(f"YOLO cache directory: {YOLO_CACHE_DIR}")
        logger.info(f"YOLO cache directory exists: {YOLO_CACHE_DIR.exists()}")
        logger.info(f"YOLO cache directory contents: {list(YOLO_CACHE_DIR.glob('*'))}")
        
        logger.info(f"Loading person detection model: {person_model_path}")
        self.person_model = self._load_yolo_model(person_model_path)
        logger.info(f"Loading face detection model: {face_model_path}")
        self.face_model = self._load_face_model(face_model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize helper processors
        self.image_resizer = ImageResizer()
        self.metadata_processor = MetadataProcessor()
        
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """Load a regular YOLO model."""
        try:
            logger.info(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            logger.info(f"Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
            
    def _load_face_model(self, model_path: str) -> YOLO:
        """Load the face detection model."""
        try:
            cache_path = YOLO_CACHE_DIR / model_path
            
            if not cache_path.exists():
                logger.info(f"Face model not found in {cache_path}, downloading...")
                url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
                
                response = requests.get(url)
                response.raise_for_status()
                
                cache_path.write_bytes(response.content)
                logger.info(f"Face model downloaded to {cache_path}")
            
            model = YOLO(str(cache_path))
            logger.info("Face model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load face model: {e}")
            raise

    def process_image(self, image: ExportedImage) -> List[Region]:
        """Process an image to detect and crop all persons and faces."""
        regions = []
        
        with Image.open(image.source_path) as img:
            original_size = img.size
        
        # YOLO face detection
        logger.info("Running YOLO face detection...")
        face_results = self.face_model(str(image.source_path), conf=self.confidence_threshold)[0]
        self._process_face_detections(face_results, regions, original_size)
        
        # YOLO person detection
        logger.info("Running YOLO person detection...")
        person_results = self.person_model(str(image.source_path), conf=self.confidence_threshold)[0]
        self._process_person_detections(person_results, regions, original_size)
        
        return regions
    
    def save_crops(self, 
                  image: ExportedImage, 
                  regions: List[Region], 
                  output_dir: Path,
                  perform_resize: bool = False,
                  add_metadata: bool = False,
                  kalliste_metadata: Optional[Dict] = None) -> List[Path]:
        """
        Save cropped regions to output directory with optional resizing and metadata.
        
        Args:
            image: ExportedImage to process
            regions: List of regions to crop
            output_dir: Directory to save crops
            perform_resize: Whether to resize to SDXL dimensions
            add_metadata: Whether to add Kalliste metadata
            kalliste_metadata: Optional Kalliste metadata to add
            
        Returns:
            List of paths to saved crop files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        with Image.open(image.source_path) as img:
            # Get original EXIF data if available
            exif_bytes = None
            if "exif" in img.info:
                try:
                    exif_dict = piexif.load(img.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                except Exception as e:
                    logger.warning(f"Could not preserve EXIF data: {e}")
            
            for i, region in enumerate(regions):
                # First, create the basic crop
                crop = img.crop((region.x1, region.y1, region.x2, region.y2))
                base_name = image.source_path.stem
                
                # Generate output path
                if region.region_type == 'face':
                    filename = f"{base_name}_face_{i}"
                else:
                    conf_str = f"{region.confidence:.2f}" if region.confidence else "unknown"
                    filename = f"{base_name}_{region.region_type}_{i}_conf{conf_str}"
                
                if perform_resize:
                    filename += "_sdxl"
                
                output_path = output_dir / f"{filename}.png"
                
                # Handle resizing if requested
                if perform_resize:
                    width, height = crop.size
                    target_dims = RegionProcessor.get_target_dimensions(width, height)
                    
                    if target_dims:
                        self.image_resizer.resize_image(
                            image=crop,
                            size=target_dims,
                            maintain_aspect=True
                        )
                
                try:
                    if add_metadata and kalliste_metadata:
                        # Save with temporary name first
                        temp_path = output_dir / f"temp_{filename}.png"
                        if exif_bytes:
                            crop.save(temp_path, format='PNG', exif=exif_bytes)
                        else:
                            crop.save(temp_path, format='PNG')
                        
                        # Add Kalliste metadata
                        self.metadata_processor.copy_metadata(
                            source_path=image.source_path,
                            dest_path=temp_path,
                            kalliste_metadata=kalliste_metadata
                        )
                        
                        # Rename to final name
                        temp_path.rename(output_path)
                    else:
                        # Save directly with EXIF if available
                        if exif_bytes:
                            crop.save(output_path, format='PNG', exif=exif_bytes)
                        else:
                            crop.save(output_path, format='PNG')
                            
                    saved_paths.append(output_path)
                    
                except Exception as e:
                    logger.warning(f"Error saving crop: {e}")
                    continue
        
        return saved_paths

    def _process_face_detections(self, results, regions, original_size):
        """Process YOLO face detection results."""
        for box in results.boxes:
            conf = float(box.conf[0])
            logger.info(f"Face detection confidence: {conf}")
            if conf >= self.confidence_threshold:
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Calculate center and dimensions
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Expand box by 40%
                expansion = 0.4
                new_width = width * (1 + expansion)
                new_height = height * (1 + expansion)
                
                # Calculate new coordinates
                x1_expanded = max(0, int(center_x - new_width/2))
                x2_expanded = min(original_size[0], int(center_x + new_width/2))
                y1_expanded = max(0, int(center_y - new_height/2))
                y2_expanded = min(original_size[1], int(center_y + new_height/2))
                
                region = Region(x1_expanded, y1_expanded, 
                              x2_expanded, y2_expanded,
                              'face', conf)
                regions.append(region)
                logger.info(f"Added face region with dimensions {width}x{height}")
    
    def _process_person_detections(self, results, regions, original_size):
        """Process YOLO person detection results."""
        person_detections = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Person class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                logger.info(f"Person detection confidence: {conf}")
                if conf >= self.confidence_threshold:
                    person_detections.append((xyxy, conf))
        
        if person_detections:
            sorted_detections = sorted(person_detections, 
                                    key=lambda d: d[1],  # Sort by confidence
                                    reverse=True)
            
            for xyxy, conf in sorted_detections:
                region = Region(
                    int(xyxy[0]), int(xyxy[1]), 
                    int(xyxy[2]), int(xyxy[3]),
                    'person', conf
                )
                regions.append(region)
                width, height = region.get_dimensions()
                logger.info(f"Added person region with dimensions {width}x{height}")