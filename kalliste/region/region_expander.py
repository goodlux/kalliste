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
        """Expand region while maintaining proportions and staying within bounds."""
        # First apply type-specific padding equally
        expand_factor = cls.EXPANSION_FACTORS.get(
            region.region_type.lower(), 
            cls.EXPANSION_FACTORS['default']
        )
        
        # Initial expansion
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

        # Find best fitting SDXL ratio
        best_fit = None
        min_boundary_overflow = float('inf')
        
        for ratio, (target_w, target_h) in cls.SDXL_RATIOS:
            # Try both fitting strategies: constrain by width or by height
            options = []
            
            # Option 1: Constrain by current width
            trial_h = width / ratio
            options.append((width, trial_h))
            
            # Option 2: Constrain by current height
            trial_w = height * ratio
            options.append((trial_w, height))
            
            # Evaluate each option
            for new_width, new_height in options:
                # Calculate boundaries
                half_w = new_width / 2
                half_h = new_height / 2
                
                # Calculate how much we'd overflow boundaries
                left_overflow = max(0, half_w - center_x)
                right_overflow = max(0, (center_x + half_w) - img_width)
                top_overflow = max(0, half_h - center_y)
                bottom_overflow = max(0, (center_y + half_h) - img_height)
                
                total_overflow = left_overflow + right_overflow + top_overflow + bottom_overflow
                
                if total_overflow < min_boundary_overflow:
                    min_boundary_overflow = total_overflow
                    best_fit = (new_width, new_height, ratio)
        
        if not best_fit:
            return expanded
            
        new_width, new_height, chosen_ratio = best_fit
        
        # Now adjust the center point if needed to avoid boundaries
        # This ensures we never go out of bounds, even if it means
        # shifting the center of the crop
        half_w = new_width / 2
        half_h = new_height / 2
        
        # Adjust center_x to avoid boundary overflow
        if center_x - half_w < 0:
            center_x = half_w
        elif center_x + half_w > img_width:
            center_x = img_width - half_w
            
        # Adjust center_y to avoid boundary overflow
        if center_y - half_h < 0:
            center_y = half_h
        elif center_y + half_h > img_height:
            center_y = img_height - half_h
        
        # Calculate final coordinates
        x1 = max(0, int(center_x - half_w))
        x2 = min(img_width, int(center_x + half_w))
        y1 = max(0, int(center_y - half_h))
        y2 = min(img_height, int(center_y + half_h))
        
        # One final adjustment to ensure we maintain exact ratio
        # even after boundary constraints
        final_width = x2 - x1
        final_height = y2 - y1
        final_ratio = final_width / final_height
        
        if final_ratio > chosen_ratio:
            # Too wide, adjust width
            target_width = int(final_height * chosen_ratio)
            x_adjust = (final_width - target_width) // 2
            x1 += x_adjust
            x2 -= x_adjust
        elif final_ratio < chosen_ratio:
            # Too tall, adjust height
            target_height = int(final_width / chosen_ratio)
            y_adjust = (final_height - target_height) // 2
            y1 += y_adjust
            y2 -= y_adjust
        
        return Region(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            region_type=region.region_type,
            confidence=region.confidence
        )