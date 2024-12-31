"""Region handling package for Kalliste.

This package contains the Region class and classes for manipulating regions (bounding boxes) 
for image processing:
- Region: Core class representing a rectangular region in an image
- RegionExpander: Expands regions to match training model ratios
- RegionDownsizer: Downsizes regions to match target pixel dimensions
- RegionMatcher: Matches and transfers data between different region types
"""

from .region import Region
from .region_expander import RegionExpander
from .region_downsizer import RegionDownsizer
from .region_matcher import RegionMatcher

__all__ = ['Region', 'RegionExpander', 'RegionDownsizer', 'RegionMatcher']