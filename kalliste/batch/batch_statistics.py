"""Handles statistics collection and tracking for image batch processing."""
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, List

@dataclass
class BatchStatistics:
    """Stores comprehensive statistics for a batch of processed images."""
    
    # Basic counts
    original_images_processed: int = 0
    
    # Region detection statistics
    regions_by_type: Counter = field(default_factory=Counter)
    small_regions_by_type: Counter = field(default_factory=Counter)
    
    # Assessment statistics by region type
    technical_assessments: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    aesthetic_assessments: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    overall_assessments: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    kalliste_assessments: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    
    def increment_original_images(self):
        """Increment count of original images processed."""
        self.original_images_processed += 1
        
    def add_region(self, region_type: str):
        """Record a detected region."""
        self.regions_by_type[region_type] += 1
        
    def add_small_region(self, region_type: str):
        """Record a region rejected for being too small."""
        self.small_regions_by_type[region_type] += 1
        
    def add_assessments(self, region_type: str, technical: str, aesthetic: str, 
                       overall: str, kalliste: str):
        """Record all assessments for a processed region."""
        self.technical_assessments[region_type][technical] += 1
        self.aesthetic_assessments[region_type][aesthetic] += 1
        self.overall_assessments[region_type][overall] += 1
        self.kalliste_assessments[region_type][kalliste] += 1
