"""Handles detection and cropping of persons and faces in images."""
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from PIL import Image
from ultralytics import YOLO

from ..models.exported_image import ExportedImage, Region


class CropProcessor:
    """Processes images to detect and crop persons, with aspect ratio adjustment for SDXL."""
    
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
    
    def __init__(self, model_path: str = 'yolov11n.pt', confidence_threshold: float = 0.3):
        """Initialize the CropProcessor with YOLO model and parameters."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
    
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
    
    def process_image(self, image: ExportedImage) -> List[Region]:
        """Process an image to detect and crop persons."""
        # Run YOLO prediction
        results = self.model.predict(
            source=str(image.source_path),
            conf=self.confidence_threshold,
            verbose=False
        )[0]  # Get first (and only) result
        
        regions = []
        
        # Get person detections (class 0)
        person_detections = []
        for box in results.boxes:
            if box.cls == 0:  # Person class
                xyxy = box.xyxy[0].cpu().numpy()  # Get box coordinates
                conf = float(box.conf[0])         # Get confidence
                if conf >= self.confidence_threshold:
                    person_detections.append((xyxy, conf))
        
        if person_detections:
            # Sort by confidence and get best detection
            best_detection = sorted(person_detections, key=lambda x: x[1], reverse=True)[0]
            xyxy, conf = best_detection
            
            # Create initial region
            region = Region(
                int(xyxy[0]), int(xyxy[1]), 
                int(xyxy[2]), int(xyxy[3]),
                'person', conf
            )
            
            # Get image size
            with Image.open(image.source_path) as img:
                original_size = img.size
            
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
        """Save cropped regions to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with Image.open(image.source_path) as img:
            for i, region in enumerate(regions):
                crop = img.crop((region.x1, region.y1, region.x2, region.y2))
                
                # Generate output filename
                base_name = image.source_path.stem
                output_path = output_dir / f"{base_name}_person_{i}.png"
                
                # Save crop
                crop.save(output_path, format='PNG')
                # TODO: Preserve metadata