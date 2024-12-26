"""Utility functions for image processing operations across Kalliste.

This module contains shared utility functions used by various image processing
components in Kalliste, including:
- Region overlap calculations (IoU)
- Common image geometry operations
- Shared image processing constants

These utilities are designed to be used by different processors (RegionProcessor,
CropProcessor, etc.) to maintain consistent behavior across the application.
"""

from typing import Union, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
from kalliste.models.exported_image import Region


def calculate_iou(region1: Region, region2: Region) -> float:
    """Calculate Intersection over Union (IoU) between two regions.
    
    IoU is calculated as the area of intersection divided by the area of union
    for two bounding boxes. This is commonly used to measure overlap between
    detected regions or to match regions from different sources.
    
    Args:
        region1: First region
        region2: Second region
        
    Returns:
        float: IoU score between 0 and 1
               0 = no overlap
               1 = perfect overlap
    """
    # Calculate intersection coordinates
    x_left = max(region1.x1, region2.x1)
    y_top = max(region1.y1, region2.y1)
    x_right = min(region1.x2, region2.x2)
    y_bottom = min(region1.y2, region2.y2)
    
    # Check if regions overlap at all
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    # Calculate intersection area
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate individual region areas
    region1_area = (region1.x2 - region1.x1) * (region1.y2 - region1.y1)
    region2_area = (region2.x2 - region2.x1) * (region2.y2 - region2.y1)
    
    # Calculate union area
    union = region1_area + region2_area - intersection
    
    # Return IoU score
    return intersection / union if union > 0 else 0.0


def center_point(region: Region) -> Tuple[float, float]:
    """Calculate the center point of a region.
    
    Args:
        region: Region to find center of
        
    Returns:
        Tuple of (center_x, center_y) coordinates
    """
    center_x = (region.x1 + region.x2) / 2
    center_y = (region.y1 + region.y2) / 2
    return (center_x, center_y)


def region_dimensions(region: Region) -> Tuple[int, int]:
    """Calculate width and height of a region.
    
    Args:
        region: Region to measure
        
    Returns:
        Tuple of (width, height) in pixels
    """
    width = region.x2 - region.x1
    height = region.y2 - region.y1
    return (width, height)


def expand_region(region: Region, 
                 expand_x: float = 0.0,
                 expand_y: float = 0.0,
                 image_size: Optional[Tuple[int, int]] = None) -> Region:
    """Expand a region by specified percentages while optionally respecting image boundaries.
    
    Args:
        region: Region to expand
        expand_x: Horizontal expansion factor (e.g., 0.4 = 40% wider)
        expand_y: Vertical expansion factor (e.g., 0.5 = 50% taller)
        image_size: Optional (width, height) of containing image to respect boundaries
        
    Returns:
        New expanded Region
    """
    width, height = region_dimensions(region)
    center_x, center_y = center_point(region)
    
    # Calculate new dimensions
    new_width = width * (1 + expand_x)
    new_height = height * (1 + expand_y)
    
    # Calculate new coordinates
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
