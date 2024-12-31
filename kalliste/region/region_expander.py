"""Handles expansion of regions to match training model ratios."""
from typing import Tuple, Optional
from .region import Region
import logging

logger = logging.getLogger(__name__)

class RegionExpander:
    """Expands regions to match training model ratios."""
    
    # SDXL dimensions and their ratios
    SDXL_DIMENSIONS = [
        ((1024, 1024), (1, 1)),      # Square
        ((1152, 896), (9, 7)),       # Landscape
        ((896, 1152), (7, 9)),       # Portrait
        ((1216, 832), (19, 13)),     # Landscape
        ((832, 1216), (13, 19)),     # Portrait
        ((1344, 768), (7, 4)),       # Landscape
        ((768, 1344), (4, 7)),       # Portrait
        ((1536, 640), (12, 5)),      # Landscape
        ((640, 1536), (5, 12))       # Portrait
    ]
    
    SDXL_RATIOS = [(w/h, (w,h)) for (w,h), _ in SDXL_DIMENSIONS]
    
    # Type-specific expansion factors
    EXPANSION_FACTORS = {
        'face': (0.4, 0.5),     # 40% horizontal, 50% vertical
        'person': (0.15, 0.1),  # 15% horizontal, 10% vertical
        'default': (0.05, 0.05) # 5% default padding
    }
    
    @staticmethod
    def center_point(region: Region) -> Tuple[float, float]:
        """Calculate the center point of a region."""
        center_x = (region.x1 + region.x2) / 2
        center_y = (region.y1 + region.y2) / 2
        return (center_x, center_y)
    
    @staticmethod
    def region_dimensions(region: Region) -> Tuple[int, int]:
        """Calculate width and height of a region."""
        width = region.x2 - region.x1
        height = region.y2 - region.y1
        return (width, height)
    
    @classmethod
    def expand_region(cls, region: Region, 
                     expand_x: float = 0.0,
                     expand_y: float = 0.0,
                     image_size: Optional[Tuple[int, int]] = None) -> Region:
        """Expand a region by specified percentages while respecting image boundaries."""
        width, height = cls.region_dimensions(region)
        center_x, center_y = cls.center_point(region)
        
        # Calculate new dimensions
        new_width = width * (1 + expand_x)
        new_height = height * (1 + expand_y)
        
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
        """Expand region to match closest SDXL ratio while maintaining center."""
        # First apply type-specific padding
        expand_x, expand_y = cls.EXPANSION_FACTORS.get(
            region.region_type.lower(), 
            cls.EXPANSION_FACTORS['default']
        )
        
        padded_region = cls.expand_region(
            region,
            expand_x=expand_x,
            expand_y=expand_y,
            image_size=(img_width, img_height)
        )
        
        # Get current dimensions
        width, height = cls.region_dimensions(padded_region)
        current_ratio = width / height
        
        # Find closest SDXL ratio
        target_ratio, _ = min(cls.SDXL_RATIOS, 
                            key=lambda x: abs(x[0] - current_ratio))
        
        # Calculate expansion factors needed to match ratio
        if current_ratio < target_ratio:
            # Need to increase width
            ratio_expand_x = (height * target_ratio / width) - 1
            ratio_expand_y = 0
        else:
            # Need to increase height
            ratio_expand_x = 0
            ratio_expand_y = (width / target_ratio / height) - 1
            
        # Apply ratio-matching expansion
        final_region = cls.expand_region(
            padded_region,
            expand_x=ratio_expand_x,
            expand_y=ratio_expand_y,
            image_size=(img_width, img_height)
        )
        
        return final_region