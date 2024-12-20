# kalliste/processors/region_processor.py
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
from PIL import Image
from kalliste.models.exported_image import Region

class RegionProcessor:
    """Handles region processing and SDXL format conversion."""
    
    # SDXL training dimensions and their ratios
    SDXL_DIMENSIONS = [
        ((1024, 1024), (1, 1)),      # Square
        ((1152, 896), (9, 7)),       # Landscape
        ((896, 1152), (7, 9)),       # Portrait
        ((1216, 832), (19, 13)),     # Landscape
        ((832, 1216), (13, 19)),     # Portrait
        ((1344, 768), (7, 4)),       # Landscape
        ((768, 1344), (4, 7)),       # Portrait
        ((1536, 640), (12, 5)),      # Landscape
        ((640, 1536), (5, 12))       # Portrait (iPhone)
    ]
    
    SDXL_RATIOS = [(w/h, (w,h)) for (w,h), _ in SDXL_DIMENSIONS]
    
    @classmethod
    def get_target_dimensions(cls, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Find the nearest SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        closest_ratio = min(cls.SDXL_RATIOS, key=lambda x: abs(x[0] - current_ratio))
        target_dims = closest_ratio[1]
        
        return None if width < target_dims[0] or height < target_dims[1] else target_dims

    @classmethod
    def expand_bbox_to_ratio(cls, bbox: Region, target_dims: Tuple[int, int],
                           original_size: Tuple[int, int], region_type: str) -> Optional[Region]:
        """Expand bounding box to match target dimensions while maintaining aspect ratio."""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        crop_width, crop_height = x2 - x1, y2 - y1
        
        target_w, target_h = target_dims
        target_ratio = target_w / target_h
        
        # Calculate new dimensions
        if (crop_width / crop_height) < target_ratio:
            new_width = int(crop_height * target_ratio)
            new_height = crop_height
        else:
            new_width = crop_width
            new_height = int(crop_width / target_ratio)
        
        if new_width < target_w or new_height < target_h:
            return None
            
        # Calculate new coordinates
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        x1_new = max(0, int(mid_x - new_width / 2))
        x2_new = min(original_size[0], int(mid_x + new_width / 2))
        y1_new = max(0, int(mid_y - new_height / 2))
        y2_new = min(original_size[1], int(mid_y + new_height / 2))
        
        region = Region(x1_new, y1_new, x2_new, y2_new, region_type, bbox.confidence)
        region.add_tag('SDXL')
        return region

    @staticmethod
    def calculate_score(detection: Tuple[np.ndarray, float]) -> float:
        """Calculate a score for detection prioritization."""
        xyxy, conf = detection
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        return conf * (area ** 0.5)