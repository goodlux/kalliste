"""Handles downsizing regions to match SDXL dimensions."""
from typing import Tuple
from .region import Region
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RegionDownsizer:
    """Downsizes regions to match Stability AI SDXL dimensions."""
    
    # Official Stability AI SDXL dimensions
    SDXL_DIMENSIONS = [
        (1024, 1024),  # square
        (1152, 896),   # landscape
        (896, 1152),   # portrait
        (1216, 832),   # wide landscape
        (1344, 768),   # wider landscape 
        (768, 1344),   # tall portrait
    ]
    
    @classmethod
    def get_target_dimensions(cls, width: int, height: int) -> Tuple[int, int]:
        """Find the matching SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        
        # Find closest matching SDXL ratio
        return min(cls.SDXL_DIMENSIONS, 
                  key=lambda dims: abs(dims[0]/dims[1] - current_ratio))
    
    @classmethod
    def is_valid_size(cls, region: Region, img: Image.Image, min_ratio: float = 0.5) -> bool:
        """Check if region is large enough to be worth processing."""
        width = region.x2 - region.x1
        height = region.y2 - region.y1
        target_dims = cls.get_target_dimensions(width, height)
        return (width >= target_dims[0] * min_ratio and 
                height >= target_dims[1] * min_ratio)
    
    @classmethod
    def downsize_to_sdxl(cls, img: Image.Image) -> Image.Image:
        """Downsize image to SDXL dimensions using high-quality thumbnail resize."""
        # Get current ratio
        current_ratio = img.width / img.height
        
        # Find target SDXL dimensions
        target_dims = min(cls.SDXL_DIMENSIONS, 
                         key=lambda dims: abs(dims[0]/dims[1] - current_ratio))
        
        # Make a copy since thumbnail modifies in place
        img_copy = img.copy()
        
        # Use thumbnail to maintain aspect ratio while fitting within target dims
        img_copy.thumbnail(target_dims, Image.Resampling.LANCZOS)
        
        return img_copy