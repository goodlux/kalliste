"""Matches named regions (like lr_faces) to detected regions."""
from typing import List, Dict, Optional, Tuple
from .region import Region
from ..tag.kalliste_tag import KallisteStringTag, KallisteBagTag
import logging

logger = logging.getLogger(__name__)

class RegionMatcher:
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize with IoU threshold for matching."""
        self.iou_threshold = iou_threshold
    
    def _iou(self, region1: Region, region2: Dict) -> float:
        """Calculate Intersection over Union between a Region and LR face dict."""
        # Convert region2's dimensions to x1,y1,x2,y2 format
        bbox = region2['bbox']
        r2_x1 = bbox['x']
        r2_y1 = bbox['y']
        r2_x2 = r2_x1 + bbox['w']
        r2_y2 = r2_y1 + bbox['h']
        
        # Calculate intersection
        x1 = max(region1.x1, r2_x1)
        y1 = max(region1.y1, r2_y1)
        x2 = min(region1.x2, r2_x2)
        y2 = min(region1.y2, r2_y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        region1_area = (region1.x2 - region1.x1) * (region1.y2 - region1.y1)
        region2_area = bbox['w'] * bbox['h']
        
        # Calculate union
        union = region1_area + region2_area - intersection
        
        if union <= 0:
            return 0.0
            
        return intersection / union

    def _find_best_face_match(self, face_region: Region, lr_faces: List[Dict], 
                             matched_faces: set) -> Tuple[Optional[Dict], Optional[int], float]:
        """Find the best matching LR face for a detected face region."""
        best_match = None
        best_iou = self.iou_threshold
        best_face_idx = None
        
        for i, face in enumerate(lr_faces):
            if i in matched_faces:
                continue
                
            iou = self._iou(face_region, face)
            logger.debug(f"IoU with face '{face['name']}': {iou:.3f}")
            
            if iou > best_iou:
                best_iou = iou
                best_match = face
                best_face_idx = i
                
        return best_match, best_face_idx, best_iou

    def _find_containing_person(self, face_region: Region, person_regions: List[Region]) -> Optional[Region]:
        """Find the person region that best contains this face region."""
        best_person = None
        best_coverage = 0.0
        
        for person in person_regions:
            # Calculate how much of the face is inside the person region
            x1 = max(face_region.x1, person.x1)
            y1 = max(face_region.y1, person.y1)
            x2 = min(face_region.x2, person.x2)
            y2 = min(face_region.y2, person.y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            intersection = (x2 - x1) * (y2 - y1)
            face_area = (face_region.x2 - face_region.x1) * (face_region.y2 - face_region.y1)
            coverage = intersection / face_area
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_person = person
                
        return best_person if best_coverage > 0.5 else None

    def match_faces(self, detected_regions: List[Region], lr_faces: List[Dict]) -> List[Region]:
        """Match detected regions to LR face metadata and establish region hierarchy."""
        if not lr_faces:
            logger.debug("No LR faces to match")
            return detected_regions
            
        logger.debug(f"DEBUG 1: Found {len(lr_faces)} LR faces to match: {lr_faces}")
        
        # Separate face and person regions
        face_regions = [r for r in detected_regions if r.region_type == 'face']
        person_regions = [r for r in detected_regions if r.region_type == 'person']
        
        logger.debug(f"DEBUG 2: Found {len(face_regions)} face regions and {len(person_regions)} person regions")
        logger.debug(f"DEBUG 3: Face regions coords: {[(r.x1,r.y1,r.x2,r.y2) for r in face_regions]}")
        logger.debug(f"DEBUG 4: Person regions coords: {[(r.x1,r.y1,r.x2,r.y2) for r in person_regions]}")
        
        matched_faces = set()
        
        for face_region in face_regions:
            logger.debug(f"DEBUG 5: Processing face region: ({face_region.x1},{face_region.y1},{face_region.x2},{face_region.y2})")
            
            best_match, best_face_idx, best_iou = self._find_best_face_match(
                face_region, lr_faces, matched_faces
            )
            
            if best_match:
                logger.info(f"DEBUG 6: Matched face to '{best_match['name']}' with IoU {best_iou:.3f}")
                name_tag = KallisteStringTag("KallistePersonName", best_match['name'])
                face_region.add_tag(name_tag)
                matched_faces.add(best_face_idx)
                
                # Verify tag was added correctly
                if face_region.has_tag("KallistePersonName"):
                    logger.debug(f"DEBUG 7: Successfully added KallistePersonName tag: {face_region.get_tag('KallistePersonName').value}")
                else:
                    logger.error("DEBUG 8: Failed to add KallistePersonName tag!")
                
                # Find containing person region
                person_region = self._find_containing_person(face_region, person_regions)
                if person_region:
                    logger.info(f"DEBUG 9: Found containing person region")
                    person_region.add_tag(name_tag)
                    
                    # Verify tag was added to person region
                    if person_region.has_tag("KallistePersonName"):
                        logger.debug(f"DEBUG 10: Added tag to person region: {person_region.get_tag('KallistePersonName').value}")
                    else:
                        logger.error("DEBUG 11: Failed to add tag to person region!")
            else:
                logger.debug("DEBUG 12: No matching face found for this region")

        # Final verification
        logger.debug("DEBUG 13: Final state of regions:")
        for region in detected_regions:
            if region.has_tag("KallistePersonName"):
                logger.debug(f"Region type {region.region_type} has name: {region.get_tag('KallistePersonName').value}")
            else:
                logger.debug(f"Region type {region.region_type} has no name tag")
        
        return detected_regions
    
 