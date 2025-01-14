"""Handles expansion of regions to match training model ratios."""
from typing import Tuple, Optional
from .region import Region
import logging

logger = logging.getLogger(__name__)

class RegionExpander:
    """Expands regions to match Stability AI SDXL training dimensions."""
    
    # Official Stability AI SDXL dimensions
    SDXL_DIMENSIONS = [
        (1024, 1024),  # square
        (1152, 896),   # landscape
        (896, 1152),   # portrait
        (1216, 832),
        (832, 1216),     # wide landscape
        (1344, 768),   # wider landscape 
        (768, 1344),   # tall portrait
    ]

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
    def expand_region_to_sdxl_ratios(cls, region: Region, img_width: int, img_height: int) -> Region:
        """Expand region to match SDXL dimensions while preserving proportions."""
        
        # Initial expansion based on content type
        expand_factor = cls.EXPANSION_FACTORS.get(
            region.region_type.lower(), 
            cls.EXPANSION_FACTORS['default']
        )
        
        # Apply initial expansion
        expanded = cls.expand_region(
            region,
            expand_factor=expand_factor,
            image_size=(img_width, img_height)
        )
        
        # Get current dimensions after expansion
        width = expanded.x2 - expanded.x1
        height = expanded.y2 - expanded.y1
        current_ratio = width / height
        center_x = (expanded.x1 + expanded.x2) / 2
        center_y = (expanded.y1 + expanded.y2) / 2

        # Find best matching SDXL dimensions
        best_dims = None
        min_ratio_diff = float('inf')
        max_center_shift = width/8  # Allow up to 1/8 width shift of center point
        
        for target_width, target_height in cls.SDXL_DIMENSIONS:
            target_ratio = target_width / target_height
            
            # Calculate dimensions that would match this ratio
            if current_ratio >= target_ratio:
                trial_width = width
                trial_height = int(width / target_ratio)
            else:
                trial_height = height
                trial_width = int(height * target_ratio)
            
            # Check if this would fit within bounds
            half_w = trial_width / 2
            half_h = trial_height / 2
            
            # Try to maintain center point
            potential_center_x = center_x
            potential_center_y = center_y
            
            # Adjust center point if needed to stay in bounds
            if center_x - half_w < 0:
                potential_center_x = half_w
            elif center_x + half_w > img_width:
                potential_center_x = img_width - half_w
                
            if center_y - half_h < 0:
                potential_center_y = half_h
            elif center_y + half_h > img_height:
                potential_center_y = img_height - half_h
            
            # Calculate how much we need to move the center
            center_shift = abs(potential_center_x - center_x) + abs(potential_center_y - center_y)
            
            # If we don't need to shift the center too much, consider this ratio
            if center_shift <= max_center_shift:
                ratio_diff = abs(current_ratio - target_ratio)
                if ratio_diff < min_ratio_diff:
                    min_ratio_diff = ratio_diff
                    best_dims = (trial_width, trial_height, potential_center_x, potential_center_y)
        
        # If no good match found with center shift constraint, try again allowing more shift
        if not best_dims:
            logger.debug("No match found with tight center constraint, allowing more shift")
            max_center_shift = width/4  # Allow up to 1/4 width shift
            
            for target_width, target_height in cls.SDXL_DIMENSIONS:
                target_ratio = target_width / target_height
                
                if current_ratio >= target_ratio:
                    trial_width = width
                    trial_height = int(width / target_ratio)
                else:
                    trial_height = height
                    trial_width = int(height * target_ratio)
                
                half_w = trial_width / 2
                half_h = trial_height / 2
                
                potential_center_x = max(half_w, min(img_width - half_w, center_x))
                potential_center_y = max(half_h, min(img_height - half_h, center_y))
                
                # Calculate center shift
                center_shift = abs(potential_center_x - center_x) + abs(potential_center_y - center_y)
                
                if center_shift <= max_center_shift:
                    ratio_diff = abs(current_ratio - target_ratio)
                    if ratio_diff < min_ratio_diff:
                        min_ratio_diff = ratio_diff
                        best_dims = (trial_width, trial_height, potential_center_x, potential_center_y)
        
        # Use best match found
        if best_dims:
            new_width, new_height, center_x, center_y = best_dims
        else:
            # Fallback to basic expansion if no good match found
            logger.warning("No suitable SDXL ratio found, using basic expansion")
            return expanded
        
        # Calculate final coordinates
        half_w = new_width / 2
        half_h = new_height / 2
        
        x1 = max(0, int(center_x - half_w))
        x2 = min(img_width, int(center_x + half_w))
        y1 = max(0, int(center_y - half_h))
        y2 = min(img_height, int(center_y + half_h))
        
        return Region(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            region_type=region.region_type,
            confidence=region.confidence
        )