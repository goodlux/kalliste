"""Handles expansion of regions to match training model ratios."""
from typing import Tuple, Optional
from .region import Region
import logging

logger = logging.getLogger(__name__)

class RegionExpander:
    """Expands regions to match Stability AI SDXL training dimensions."""
    
    SDXL_DIMENSIONS = [
        (1024, 1024),  # square
        (1152, 896),   # landscape
        (896, 1152),   # portrait
        (1216, 832),   # wide landscape
        (832, 1216),   # portrait
        (1344, 768),   # wider landscape 
        (768, 1344),   # tall portrait
    ]

    EXPANSION_FACTORS = {
        'face': 0.4,    
        'person': 0.15,
        'default': 0.05
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
    def get_target_sdxl_dimensions(cls, width: int, height: int) -> tuple[int, int]:
        """Find the matching SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        return min(cls.SDXL_DIMENSIONS, 
                  key=lambda dims: abs(dims[0]/dims[1] - current_ratio))

    @classmethod
    def expand_region_to_sdxl_ratios(cls, region: Region, img_width: int, img_height: int) -> Optional[Region]:
        """Expand region to match SDXL dimensions."""
        expand_factor = cls.EXPANSION_FACTORS.get(
            region.region_type.lower(), 
            cls.EXPANSION_FACTORS['default']
        )
        
        width = region.x2 - region.x1
        height = region.y2 - region.y1
        
        # Get target SDXL dimensions
        target_width, target_height = cls.get_target_sdxl_dimensions(width, height)
        target_ratio = target_width / target_height
        
        logger.debug(f"Original dimensions: {width}x{height}, target ratio: {target_ratio}")
        
        # Calculate dimensions that preserve target ratio after expansion
        if width / height >= target_ratio:
            new_width = width * (1 + expand_factor)
            new_height = new_width / target_ratio
        else:
            new_height = height * (1 + expand_factor)
            new_width = new_height * target_ratio
            
        center_x = (region.x1 + region.x2) / 2
        center_y = (region.y1 + region.y2) / 2
        
        # Calculate initial coordinates
        x1 = center_x - (new_width / 2)
        x2 = center_x + (new_width / 2)
        y1 = center_y - (new_height / 2)
        y2 = center_y + (new_height / 2)
        
        # Adjust if outside image bounds while maintaining ratio
        if x1 < 0:
            shift = -x1
            x1 = 0
            x2 = new_width
        elif x2 > img_width:
            shift = img_width - x2
            x2 = img_width
            x1 = x2 - new_width
            
        if y1 < 0:
            shift = -y1
            y1 = 0
            y2 = new_height
        elif y2 > img_height:
            shift = img_height - y2
            y2 = img_height
            y1 = y2 - new_height
            
        # Final bounds check
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            logger.warning(f"Could not fit expanded region within image bounds while maintaining ratio {target_ratio}")
            return None
            
        # Verify final ratio
        final_width = x2 - x1
        final_height = y2 - y1
        final_ratio = final_width / final_height
        
        if abs(final_ratio - target_ratio) > 1e-6:
            logger.warning(f"Failed to achieve target ratio: got {final_ratio}, wanted {target_ratio}")
            return None
            
        return Region(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            region_type=region.region_type,
            confidence=region.confidence
        )