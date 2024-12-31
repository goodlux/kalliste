"""Handles downsizing regions to match training model pixel dimensions."""
from typing import Tuple, Optional
from .region import Region
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RegionDownsizer:
    """Downsizes regions to match training model pixel dimensions."""
    
    # SDXL dimensions 
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
    
    @classmethod
    def get_target_dimensions(cls, width: int, height: int) -> Tuple[int, int]:
        """Find the matching SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        closest_ratio = min(cls.SDXL_RATIOS, 
                          key=lambda x: abs(x[0] - current_ratio))
        return closest_ratio[1]
        
    @classmethod
    def is_valid_size(cls, region: Region, img: Image.Image, min_ratio: float = 0.5) -> bool:
        """Check if region is large enough to be worth processing."""
        width = region.x2 - region.x1
        height = region.y2 - region.y1
        
        # Get target dimensions for this aspect ratio
        target_dims = cls.get_target_dimensions(width, height)
        
        # Check if region is at least min_ratio of target size
        return (width >= target_dims[0] * min_ratio and 
                height >= target_dims[1] * min_ratio)
        
    @classmethod
    def downsize_to_sdxl(cls, img: Image.Image, target_dims: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Downsize image to SDXL dimensions while preserving aspect ratio."""
        if target_dims is None:
            target_dims = cls.get_target_dimensions(*img.size)
            
        # Calculate resize dimensions preserving aspect ratio
        current_ratio = img.size[0] / img.size[1]
        target_ratio = target_dims[0] / target_dims[1]
        
        if current_ratio > target_ratio:
            # Width is limiting factor
            new_width = target_dims[0]
            new_height = int(new_width / current_ratio)
        else:
            # Height is limiting factor
            new_height = target_dims[1]
            new_width = int(new_height * current_ratio)
            
        # Resize with high quality settings
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create final image with padding to reach target dimensions
        final = Image.new('RGB', target_dims, (0, 0, 0))
        paste_x = (target_dims[0] - new_width) // 2
        paste_y = (target_dims[1] - new_height) // 2
        final.paste(resized, (paste_x, paste_y))
        
        return final