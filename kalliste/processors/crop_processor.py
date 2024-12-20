"""Handles detection and cropping of persons and faces in images."""
from pathlib import Path
from typing import List
import piexif
from PIL import Image
from ultralytics import YOLO

from kalliste.models.exported_image import ExportedImage, Region
from kalliste.processors.region_processor import RegionProcessor
from config.config import YOLO_WEIGHTS

class CropProcessor:
    def __init__(self, 
                 person_model_path: str = str(YOLO_WEIGHTS['object']),
                 face_model_path: str = str(YOLO_WEIGHTS['face']),
                 confidence_threshold: float = 0.4,
                 iou_threshold: float = 0.7):
        """Initialize the CropProcessor with both person and face YOLO models."""
        print(f"Loading person detection model from: {person_model_path}")
        self.person_model = YOLO(person_model_path)
        print(f"Loading face detection model from: {face_model_path}")
        self.face_model = YOLO(face_model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def process_image(self, image: ExportedImage) -> List[Region]:
        """Process an image to detect and crop all persons and faces."""
        regions = []
        
        with Image.open(image.source_path) as img:
            original_size = img.size
        
        # YOLO face detection
        print("Running YOLO face detection...")
        face_results = self.face_model(str(image.source_path), conf=self.confidence_threshold)[0]
        self._process_face_detections(face_results, regions, original_size)
        
        # YOLO person detection
        print("Running YOLO person detection...")
        person_results = self.person_model(str(image.source_path), conf=self.confidence_threshold)[0]
        self._process_person_detections(person_results, regions, original_size)
        
        return regions
    
    def _process_face_detections(self, results, regions, original_size):
        """Process YOLO face detection results."""
        for box in results.boxes:
            conf = float(box.conf[0])
            print(f"Face detection confidence: {conf}")
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
                
                width, height = region.get_dimensions()
                target_dims = RegionProcessor.get_target_dimensions(width, height)
                if target_dims:
                    expanded_region = RegionProcessor.expand_bbox_to_ratio(
                        region, target_dims, original_size, 'face')
                    if expanded_region:
                        regions.append(expanded_region)
                        print(f"Added face region with dimensions {width}x{height}")
    
    def _process_person_detections(self, results, regions, original_size):
        """Process YOLO person detection results."""
        person_detections = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Person class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                print(f"Person detection confidence: {conf}")
                if conf >= self.confidence_threshold:
                    person_detections.append((xyxy, conf))
        
        if person_detections:
            sorted_detections = sorted(person_detections, 
                                    key=RegionProcessor.calculate_score, 
                                    reverse=True)
            
            for xyxy, conf in sorted_detections:
                region = Region(
                    int(xyxy[0]), int(xyxy[1]), 
                    int(xyxy[2]), int(xyxy[3]),
                    'person', conf
                )
                width, height = region.get_dimensions()
                target_dims = RegionProcessor.get_target_dimensions(width, height)
                if target_dims:
                    expanded_region = RegionProcessor.expand_bbox_to_ratio(
                        region, target_dims, original_size, 'person')
                    if expanded_region:
                        regions.append(expanded_region)
                        print(f"Added person region with dimensions {width}x{height}")

    def save_crops(self, image: ExportedImage, regions: List[Region], output_dir: Path) -> None:
        """Save cropped regions to output directory with preserved EXIF data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with Image.open(image.source_path) as img:
            exif_bytes = None
            if "exif" in img.info:
                try:
                    exif_dict = piexif.load(img.info["exif"])
                    exif_bytes = piexif.dump(exif_dict)
                except Exception as e:
                    print(f"Warning: Could not preserve EXIF data: {e}")
            
            for i, region in enumerate(regions):
                crop = img.crop((region.x1, region.y1, region.x2, region.y2))
                base_name = image.source_path.stem
                
                if region.region_type == 'face':
                    output_path = output_dir / f"{base_name}_face_{i}_sdxl.png"
                else:
                    conf_str = f"{region.confidence:.2f}" if region.confidence else "unknown"
                    output_path = output_dir / f"{base_name}_{region.region_type}_{i}_conf{conf_str}_sdxl.png"
                
                try:
                    if exif_bytes:
                        crop.save(output_path, format='PNG', exif=exif_bytes)
                    else:
                        crop.save(output_path, format='PNG')
                except Exception as e:
                    print(f"Warning: Error saving crop: {e}")
