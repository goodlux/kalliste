"""Handles detection and cropping of persons and faces in images."""
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from PIL import Image
import piexif
from ultralytics import YOLO
import numpy as np

from kalliste.models.exported_image import ExportedImage, Region


class CropProcessor:
    """Processes images to detect and crop persons and faces, with aspect ratio adjustment for SDXL."""
    
    # Standard aspect ratios for SDXL
    SDXL_RATIOS = [
        (1, 1),    # Square
        (7, 9),    # Portrait
        (13, 19),  # Portrait
        (4, 7),    # Portrait
        (9, 7),    # Landscape
        (13, 19),  # Portrait
        (19, 13),  # Landscape
        (7, 4),    # Landscape
    ]
    
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.4,
                 iou_threshold: float = 0.7):
        """Initialize the CropProcessor with YOLO model and parameters.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def get_nearest_larger_aspect_ratio(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Find the nearest larger aspect ratio from SDXL standards."""
        current_ratio = width / height
        larger_ratios = [ratio for ratio in self.SDXL_RATIOS if ratio[0]/ratio[1] > current_ratio]
        if not larger_ratios:
            return None
        return min(larger_ratios, key=lambda r: abs(r[0]/r[1] - current_ratio))
    
    def expand_bbox_to_ratio(self, bbox: Region, target_ratio: Tuple[int, int],
                           original_size: Tuple[int, int]) -> Region:
        """Expand bounding box to match target aspect ratio."""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        crop_width, crop_height = x2 - x1, y2 - y1
        
        # Calculate target dimensions
        target_w, target_h = target_ratio
        target_ratio_float = target_w / target_h
        
        # Determine new dimensions
        if (crop_width / crop_height) < target_ratio_float:
            new_width = int(crop_height * target_ratio_float)
            new_height = crop_height
        else:
            new_width = crop_width
            new_height = int(crop_width / target_ratio_float)
            
        # Calculate new coordinates maintaining center
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        x1_new = max(0, int(mid_x - new_width / 2))
        x2_new = min(original_size[0], int(mid_x + new_width / 2))
        y1_new = max(0, int(mid_y - new_height / 2))
        y2_new = min(original_size[1], int(mid_y + new_height / 2))
        
        return Region(x1_new, y1_new, x2_new, y2_new, 'person', bbox.confidence)

    def calculate_score(self, detection: Tuple[np.ndarray, float]) -> float:
        """Calculate a score for detection prioritization.
        
        Args:
            detection: Tuple of (xyxy, confidence)
            
        Returns:
            float: Score combining size and confidence
        """
        xyxy, conf = detection
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        return conf * (area ** 0.5)  # Scale confidence by sqrt of area
    
    def process_image(self, image: ExportedImage) -> List[Region]:
        """Process an image to detect and crop all persons and faces.
        
        Returns:
            List[Region]: List of detected and processed regions (persons and faces)
        """
        regions = []
        
        # First, get face regions from Lightroom metadata
        # These are already validated for SDXL size in ExportedImage.load_face_regions
        regions.extend(image.face_regions)
        
        # Run YOLO prediction for person detection
        results = self.model(str(image.source_path), conf=self.confidence_threshold)[0]
        
        # Filter detections for person class (class 0 in COCO dataset)
        person_detections = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Person class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    person_detections.append((xyxy, conf))
        
        if person_detections:
            # Sort detections by score (combination of confidence and size)
            sorted_detections = sorted(person_detections, 
                                    key=self.calculate_score, 
                                    reverse=True)
            
            # Get image size for aspect ratio calculations
            with Image.open(image.source_path) as img:
                original_size = img.size
            
            # Process each person detection
            for xyxy, conf in sorted_detections:
                # Create initial region
                region = Region(
                    int(xyxy[0]), int(xyxy[1]), 
                    int(xyxy[2]), int(xyxy[3]),
                    'person', conf
                )
                
                # Find nearest larger aspect ratio
                crop_width, crop_height = region.get_dimensions()
                nearest_ratio = self.get_nearest_larger_aspect_ratio(crop_width, crop_height)
                
                if nearest_ratio:
                    # Expand bbox to match aspect ratio
                    expanded_region = self.expand_bbox_to_ratio(region, nearest_ratio, original_size)
                    regions.append(expanded_region)
                else:
                    regions.append(region)
        
        return regions
    
    def save_crops(self, image: ExportedImage, regions: List[Region], output_dir: Path) -> None:
        """Save cropped regions to output directory with preserved EXIF data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with Image.open(image.source_path) as img:
            # Get original EXIF data if available
            exif_bytes = None
            if "exif" in img.info:
                try:
                    exif_dict = piexif.load(img.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                except Exception as e:
                    print(f"Warning: Could not preserve EXIF data: {e}")
            
            for i, region in enumerate(regions):
                # Skip faces that don't meet SDXL size requirements
                if region.region_type == 'face' and not region.is_valid_sdxl_size():
                    continue
                
                # Crop the image
                crop = img.crop((region.x1, region.y1, region.x2, region.y2))
                
                # Generate output filename
                base_name = image.source_path.stem
                region_type = region.region_type
                
                if region.region_type == 'face':
                    # For faces, use the name if available
                    name_part = f"_{region.name}" if region.name else f"_{i}"
                    output_path = output_dir / f"{base_name}_face{name_part}.png"
                else:
                    # For persons, include confidence score
                    conf_str = f"{region.confidence:.2f}" if region.confidence is not None else "unknown"
                    output_path = output_dir / f"{base_name}_{region_type}_{i}_conf{conf_str}.png"
                
                # Save with EXIF if available
                if exif_bytes:
                    try:
                        crop.save(output_path, format='PNG', exif=exif_bytes)
                    except Exception as e:
                        print(f"Warning: Could not save with EXIF, saving without: {e}")
                        crop.save(output_path, format='PNG')
                else:
                    crop.save(output_path, format='PNG')
