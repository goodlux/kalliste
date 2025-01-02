"""Matches named regions (like lr_faces) to detected regions."""
from typing import List, Dict, Optional
from .region import Region
import logging

logger = logging.getLogger(__name__)

class RegionMatcher:
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize with IoU threshold for matching."""
        self.iou_threshold = iou_threshold
    
    def _iou(self, region1: Region, region2: Dict) -> float:
        """Calculate Intersection over Union between a Region and LR face dict."""
        # Convert LR bbox format to region coordinates
        bbox = region2['bbox']
        lr_x1 = bbox['x'] - (bbox['w'] / 2)
        lr_y1 = bbox['y'] - (bbox['h'] / 2)
        lr_x2 = lr_x1 + bbox['w']
        lr_y2 = lr_y1 + bbox['h']
        
        # Calculate intersection
        x1 = max(region1.x1, lr_x1)
        y1 = max(region1.y1, lr_y1)
        x2 = min(region1.x2, lr_x2)
        y2 = min(region1.y2, lr_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        region1_area = (region1.x2 - region1.x1) * (region1.y2 - region1.y1)
        region2_area = bbox['w'] * bbox['h']
        
        # Calculate union
        union = region1_area + region2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def match_faces(self, detected_regions: List[Region], lr_faces: List[Dict]) -> List[Region]:
        """Match detected regions to LR face metadata using IoU.
        
        Args:
            detected_regions: List of Region objects from YOLO detection
            lr_faces: List of face dictionaries from LR metadata with format:
                     {'name': str, 'bbox': {'x': float, 'y': float, 'w': float, 'h': float}}
        
        Returns:
            List of Region objects, with matched regions having name attribute set
        """
        if not lr_faces:
            return detected_regions
            
        # Keep track of which faces we've matched
        matched_faces = set()
        
        # For each detected region, find best matching LR face
        for region in detected_regions:
            if region.region_type != 'face':
                continue
                
            best_iou = self.iou_threshold
            best_match = None
            best_face_idx = None
            
            # Find best matching unmatched face
            for i, face in enumerate(lr_faces):
                if i in matched_faces:
                    continue
                    
                iou = self._iou(region, face)
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
                    best_face_idx = i
            
            # If we found a match, add the name to the region
            if best_match:
                region.name = best_match['name']
                matched_faces.add(best_face_idx)
                logger.info(f"Matched region to face: {best_match['name']} (IoU: {best_iou:.2f})")
        
        return detected_regions