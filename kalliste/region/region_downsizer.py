"""Handles downsizing regions to match training model pixel dimensions."""
from typing import Tuple, Optional
from .region import Region
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RegionDownsizer:
    """Downsizes regions to match training model pixel dimensions."""
    
    # SDXL dimensions with verified aspect ratios
    SDXL_DIMENSIONS = [
        (1024, 1024),  # 1:1
        (896, 1152),   # 3:4 (portrait)
        (1152, 896),   # 4:3 (landscape)
        (832, 1216),   # 2:3 (portrait)
        (1216, 832),   # 3:2 (landscape)
        (768, 1344),   # 1:2 (portrait)
        (1344, 768),   # 2:1 (landscape)
    ]
    
    @classmethod
    def get_target_dimensions(cls, width: int, height: int) -> Tuple[int, int]:
        """Find the matching SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        closest_dims = min(cls.SDXL_DIMENSIONS, 
                         key=lambda dims: abs(dims[0]/dims[1] - current_ratio))
        return closest_dims
        
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
        """Downsize image to SDXL dimensions using high-quality resize."""
        target_dims = cls.get_target_dimensions(*img.size)
        return img.resize(target_dims, Image.Resampling.LANCZOS)