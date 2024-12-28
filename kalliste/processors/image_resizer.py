from pathlib import Path
from typing import Dict, List, Tuple, Union
from PIL import Image
import uuid
import logging

from ..processors.metadata_processor import MetadataProcessor
from ..types import TagResult

logger = logging.getLogger(__name__)

class SDXLResizer:
    """Handles resizing images to SDXL dimensions while maintaining aspect ratio."""
    
    # SDXL dimensions from RegionProcessor
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
    def get_target_dimensions(cls, width: int, height: int) -> Tuple[int, int]:
        """Find the matching SDXL dimensions based on aspect ratio."""
        current_ratio = width / height
        closest_ratio = min(cls.SDXL_RATIOS, key=lambda x: abs(x[0] - current_ratio))
        return closest_ratio[1]
    
    @classmethod
    def resize_image(cls, image: Union[Path, Image.Image], output_path: Path) -> None:
        """Resize image to appropriate SDXL dimensions.
        
        Args:
            image: PIL Image or path to image
            output_path: Where to save the resized image
        """
        # Handle input type
        if isinstance(image, Path):
            img = Image.open(image)
        else:
            img = image
            
        try:
            # Get target dimensions
            target_dims = cls.get_target_dimensions(*img.size)
            
            # Resize with high quality settings
            resized = img.resize(target_dims, Image.Resampling.LANCZOS)
            
            # Save with PNG optimization
            resized.save(output_path, "PNG", optimize=True)
            
        finally:
            if isinstance(image, Path):
                img.close()