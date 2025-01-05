"""Handles expansion of regions to match training model ratios."""
from typing import Tuple, Optional
from .region import Region
import logging

logger = logging.getLogger(__name__)

class RegionExpander:
    """Expands regions to match training model ratios."""
    
    # SDXL dimensions with common aspect ratios
    SDXL_DIMENSIONS = [
        (1024, 1024),  # 1:1
        (896, 1152),   # 3:4 (portrait)
        (1152, 896),   # 4:3 (landscape)
        (832, 1216),   # 2:3 (portrait)
        (1216, 832),   # 3:2 (landscape)
        (768, 1344),   # 1:2 (portrait)
        (1344, 768),   # 2:1 (landscape)
    ]
    
    # Calculate ratios once
    SDXL_RATIOS = [(w/h, (w,h)) for w, h in SDXL_DIMENSIONS]
    
    # Single expansion factor based on region type
    EXPANSION_FACTORS = {
        'face': 0.4,    # 40% expansion in all directions
        'person': 0.15, # 15% expansion in all directions
        'default': 0.05 # 5% default expansion
    }
    
    @staticmethod
    def get_dimensions(region: Region) -> tuple[int, int]:
        """Calculate width and height of a region."""
        return (region.x2 - region.x1, region.y2 - region.y1)
        
    @staticmethod
    def get_center_point(region: Region) -> tuple[float, float]:
        """Calculate the center point of a region."""
        return ((region.x1 + region.x2) / 2, (region.y1 + region.y2) / 2)
    
    @classmethod
    def expand_region(cls, region: Region, 
                     expand_factor: float = 0.0,
                     image_size: Optional[Tuple[int, int]] = None) -> Region:
        """Expand a region equally in all directions, respecting image boundaries."""
        width, height = cls.get_dimensions(region)
        center_x, center_y = cls.get_center_point(region)
        
        # Calculate new dimensions with equal expansion
        new_width = width * (1 + expand_factor)
        new_height = height * (1 + expand_factor)
        
        # Calculate new coordinates from center
        x1 = center_x - (new_width / 2)
        x2 = center_x + (new_width / 2)
        y1 = center_y - (new_height / 2)
        y2 = center_y + (new_height / 2)
        
        # Constrain to image boundaries if size provided
        if image_size:
            img_width, img_height = image_size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
        
        return Region(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            region_type=region.region_type,
            confidence=region.confidence
        )

    @classmethod
    def expand_region_to_sdxl_ratios(cls, region: Region, 
                                   img_width: int, img_height: int) -> Region:
        """Expand region with equal padding, then adjust to nearest SDXL ratio within bounds."""
        # First apply type-specific padding equally
        expand_factor = cls.EXPANSION_FACTORS.get(
            region.region_type.lower(), 
            cls.EXPANSION_FACTORS['default']
        )
        
        expanded = cls.expand_region(
            region,
            expand_factor=expand_factor,
            image_size=(img_width, img_height)
        )
        
        # Get current dimensions after expansion
        width, height = cls.get_dimensions(expanded)  # Fixed method name
        current_ratio = width / height
        
        # Find closest SDXL ratio that fits within image bounds
        valid_ratios = []
        center_x, center_y = cls.get_center_point(expanded)  # Fixed method name
        
        for ratio, dims in cls.SDXL_RATIOS:
            # Calculate required dimensions to match this ratio
            if current_ratio < ratio:
                # Would need to increase width
                new_width = height * ratio
                new_height = height
            else:
                # Would need to increase height
                new_width = width
                new_height = width / ratio
            
            # Check if these dimensions would fit in image
            half_width = new_width / 2
            half_height = new_height / 2
            
            if (center_x - half_width >= 0 and 
                center_x + half_width <= img_width and
                center_y - half_height >= 0 and 
                center_y + half_height <= img_height):
                valid_ratios.append((ratio, abs(ratio - current_ratio)))
        
        if not valid_ratios:
            # If no ratio fits perfectly, return the expanded region as is
            return expanded
            
        # Use the closest valid ratio
        best_ratio = min(valid_ratios, key=lambda x: x[1])[0]
        
        # Adjust dimensions to match ratio while staying within bounds
        if current_ratio < best_ratio:
            new_width = height * best_ratio
            new_height = height
        else:
            new_width = width
            new_height = width / best_ratio
            
        # Create final region centered on original
        x1 = center_x - (new_width / 2)
        x2 = center_x + (new_width / 2)
        y1 = center_y - (new_height / 2)
        y2 = center_y + (new_height / 2)
        
        # Final boundary check
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return Region(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            region_type=region.region_type,
            confidence=region.confidence
        )