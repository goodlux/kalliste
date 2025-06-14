#!/usr/bin/env python3
"""
Advanced selection script for creating diverse, high-quality training datasets.
Creates two sets of ~5000 images each: photographs and video stills.

Uses clustering on embeddings for diversity and NIMA scores for quality selection.

Usage:
    python select_diverse_training_sets.py --target-count 5000 --min-nima-score 6.0
    python select_diverse_training_sets.py --target-count 3000 --min-nima-score 5.5 --dry-run
"""
import sys
import os
from pathlib import Path
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB
from kalliste.config import KALLISTE_DATA_DIR

# For clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("Warning: sklearn not available. Install with: pip install scikit-learn")
    HAS_SKLEARN = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageCandidate:
    """Container for image data with embeddings."""
    id: int
    image_file_path: str
    source_type: str
    nima_score_calc_average: float
    lr_rating: int
    person_name: str
    photoshoot: str
    embedding: List[float]
    
    @property
    def quality_score(self) -> float:
        """Combined quality score for ranking."""
        # Weight NIMA score more heavily, but include LR rating
        nima_weight = 0.8
        lr_weight = 0.2
        lr_normalized = self.lr_rating / 5.0  # Normalize LR rating to 0-1
        return (nima_weight * self.nima_score_calc_average) + (lr_weight * lr_normalized)

class DiverseImageSelector:
    """Selects diverse, high-quality images using clustering and quality metrics."""
    
    def __init__(self, db: MilvusDB):
        self.db = db
        
    def get_candidates(self, source_type: str, min_nima_score: float = 5.0, 
                      limit: int = None) -> List[ImageCandidate]:
        """Get candidate images of specified source type with embeddings."""
        
        # Build filter expression
        filter_expr = f'source_type == "{source_type}"'
        if min_nima_score > 0:
            filter_expr += f' && nima_score_calc_average >= {min_nima_score}'
            
        # Get all needed fields including embeddings
        output_fields = [
            "id", "image_file_path", "source_type", "nima_score_calc_average", 
            "lr_rating", "person_name", "photoshoot", "openclip_vector"
        ]
        
        logger.info(f"Querying for {source_type} images with filter: {filter_expr}")
        
        # Query the database
        results = self.db.query(
            filter_expr=filter_expr,
            limit=limit,
            output_fields=output_fields
        )
        
        if not results:
            logger.warning(f"No results found for {source_type}")
            return []
            
        # Convert to ImageCandidate objects
        candidates = []
        for result in results:
            try:
                candidate = ImageCandidate(
                    id=result.get('id', 0),
                    image_file_path=result.get('image_file_path', ''),
                    source_type=result.get('source_type', ''),
                    nima_score_calc_average=result.get('nima_score_calc_average', 0.0),
                    lr_rating=result.get('lr_rating', 0),
                    person_name=result.get('person_name', ''),
                    photoshoot=result.get('photoshoot', ''),
                    embedding=result.get('openclip_vector', [])
                )
                
                # Validate embedding exists
                if candidate.embedding and len(candidate.embedding) > 0:
                    candidates.append(candidate)
                else:
                    logger.warning(f"Skipping image {candidate.id} - missing embedding")
                    
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                continue
                
        logger.info(f"Found {len(candidates)} valid {source_type} candidates")
        return candidates
    
    def cluster_and_select(self, candidates: List[ImageCandidate], 
                          target_count: int) -> List[ImageCandidate]:
        """Use clustering to select diverse, high-quality images."""
        
        if not candidates:
            return []
            
        if not HAS_SKLEARN:
            logger.error("sklearn required for clustering. Falling back to top images.")
            return sorted(candidates, key=lambda x: x.quality_score, reverse=True)[:target_count]
            
        if len(candidates) <= target_count:
            logger.info(f"Only {len(candidates)} candidates available, returning all")
            return candidates
            
        # Extract embeddings
        embeddings = np.array([c.embedding for c in candidates])
        logger.info(f"Clustering {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Normalize embeddings for better clustering
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Determine number of clusters
        # Use more clusters than target to allow quality-based selection within clusters
        n_clusters = min(len(candidates), int(target_count * 1.5))
        
        logger.info(f"Using {n_clusters} clusters to select {target_count} images")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_normalized)
        
        # Select best images from each cluster
        selected_images = []
        cluster_stats = {}
        
        for cluster_id in range(n_clusters):
            # Get candidates in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_candidates = [candidates[i] for i in np.where(cluster_mask)[0]]
            
            if not cluster_candidates:
                continue
                
            # Sort by quality score
            cluster_candidates.sort(key=lambda x: x.quality_score, reverse=True)
            
            # Take the best candidate from this cluster
            best_candidate = cluster_candidates[0]
            selected_images.append(best_candidate)
            
            # Track cluster statistics
            cluster_stats[cluster_id] = {
                'size': len(cluster_candidates),
                'best_nima': best_candidate.nima_score_calc_average,
                'best_quality': best_candidate.quality_score,
                'selected_id': best_candidate.id
            }
        
        # Sort selected images by quality and take top target_count
        selected_images.sort(key=lambda x: x.quality_score, reverse=True)
        final_selection = selected_images[:target_count]
        
        logger.info(f"Selected {len(final_selection)} images from {n_clusters} clusters")
        
        # Log some statistics
        if final_selection:
            scores = [img.quality_score for img in final_selection]
            nima_scores = [img.nima_score_calc_average for img in final_selection]
            logger.info(f"Quality score range: {min(scores):.2f} - {max(scores):.2f}")
            logger.info(f"NIMA score range: {min(nima_scores):.2f} - {max(nima_scores):.2f}")
        
        return final_selection

def create_selection_report(photo_selection: List[ImageCandidate], 
                           video_selection: List[ImageCandidate],
                           output_dir: str) -> str:
    """Create a detailed report of the selection process."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"selection_report_{timestamp}.json")
    
    # Create comprehensive report data
    report_data = {
        "selection_timestamp": timestamp,
        "photographs": {
            "count": len(photo_selection),
            "source_type": "photograph",
            "quality_stats": _get_quality_stats(photo_selection),
            "person_distribution": _get_person_distribution(photo_selection),
            "photoshoot_distribution": _get_photoshoot_distribution(photo_selection),
            "images": [_image_to_dict(img) for img in photo_selection]
        },
        "video_stills": {
            "count": len(video_selection),
            "source_type": "video_still",
            "quality_stats": _get_quality_stats(video_selection),
            "person_distribution": _get_person_distribution(video_selection),
            "photoshoot_distribution": _get_photoshoot_distribution(video_selection),
            "images": [_image_to_dict(img) for img in video_selection]
        }
    }
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Selection report saved to: {report_path}")
    return report_path

def _get_quality_stats(images: List[ImageCandidate]) -> Dict:
    """Calculate quality statistics for a set of images."""
    if not images:
        return {}
    
    nima_scores = [img.nima_score_calc_average for img in images]
    lr_ratings = [img.lr_rating for img in images]
    quality_scores = [img.quality_score for img in images]
    
    return {
        "nima_average": np.mean(nima_scores),
        "nima_min": np.min(nima_scores),
        "nima_max": np.max(nima_scores),
        "lr_average": np.mean(lr_ratings),
        "quality_average": np.mean(quality_scores),
        "quality_min": np.min(quality_scores),
        "quality_max": np.max(quality_scores)
    }

def _get_person_distribution(images: List[ImageCandidate]) -> Dict:
    """Get distribution of images by person."""
    person_counts = {}
    for img in images:
        person = img.person_name or "Unknown"
        person_counts[person] = person_counts.get(person, 0) + 1
    return person_counts

def _get_photoshoot_distribution(images: List[ImageCandidate]) -> Dict:
    """Get distribution of images by photoshoot."""
    shoot_counts = {}
    for img in images:
        shoot = img.photoshoot or "Unknown"
        shoot_counts[shoot] = shoot_counts.get(shoot, 0) + 1
    return shoot_counts

def _image_to_dict(img: ImageCandidate) -> Dict:
    """Convert ImageCandidate to dictionary for JSON serialization."""
    return {
        "id": img.id,
        "image_file_path": img.image_file_path,
        "nima_score": img.nima_score_calc_average,
        "lr_rating": img.lr_rating,
        "quality_score": img.quality_score,
        "person_name": img.person_name,
        "photoshoot": img.photoshoot
    }

def main():
    parser = argparse.ArgumentParser(description="Select diverse training datasets from Milvus")
    parser.add_argument("--target-count", type=int, default=5000, 
                        help="Target number of images per category")
    parser.add_argument("--min-nima-score", type=float, default=5.0,
                        help="Minimum NIMA score threshold")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output directory (default: {os.path.join(KALLISTE_DATA_DIR, 'training_sets')})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show selection statistics without copying files")
    parser.add_argument("--photo-source-type", default="photograph",
                        help="Source type for photographs")
    parser.add_argument("--video-source-type", default="video_still", 
                        help="Source type for video stills")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(KALLISTE_DATA_DIR, "training_sets")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize database connection
    logger.info("Connecting to Milvus database...")
    db = MilvusDB()
    selector = DiverseImageSelector(db)
    
    try:
        # Select photographs
        logger.info(f"=== Selecting {args.target_count} photographs ===")
        photo_candidates = selector.get_candidates(
            source_type=args.photo_source_type,
            min_nima_score=args.min_nima_score
        )
        
        photo_selection = selector.cluster_and_select(photo_candidates, args.target_count)
        
        # Select video stills
        logger.info(f"=== Selecting {args.target_count} video stills ===")
        video_candidates = selector.get_candidates(
            source_type=args.video_source_type, 
            min_nima_score=args.min_nima_score
        )
        
        video_selection = selector.cluster_and_select(video_candidates, args.target_count)
        
        # Create output directories
        photo_dir = os.path.join(args.output_dir, f"photographs_{timestamp}")
        video_dir = os.path.join(args.output_dir, f"video_stills_{timestamp}")
        
        if not args.dry_run:
            os.makedirs(photo_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
        
        # Log results
        logger.info(f"=== SELECTION RESULTS ===")
        logger.info(f"Photographs selected: {len(photo_selection)}")
        logger.info(f"Video stills selected: {len(video_selection)}")
        
        if photo_selection:
            photo_nima = [img.nima_score_calc_average for img in photo_selection]
            logger.info(f"Photo NIMA range: {min(photo_nima):.2f} - {max(photo_nima):.2f}")
            
        if video_selection:
            video_nima = [img.nima_score_calc_average for img in video_selection]
            logger.info(f"Video NIMA range: {min(video_nima):.2f} - {max(video_nima):.2f}")
        
        # Create selection report
        if not args.dry_run:
            create_selection_report(photo_selection, video_selection, args.output_dir)
        
        # Export images
        if not args.dry_run:
            if photo_selection:
                logger.info(f"Exporting {len(photo_selection)} photographs to {photo_dir}")
                photo_results = [{"image_file_path": img.image_file_path} for img in photo_selection]
                db.export_images(photo_results, photo_dir)
                
            if video_selection:
                logger.info(f"Exporting {len(video_selection)} video stills to {video_dir}")
                video_results = [{"image_file_path": img.image_file_path} for img in video_selection]
                db.export_images(video_results, video_dir)
                
            logger.info("=== EXPORT COMPLETE ===")
            logger.info(f"Photographs: {photo_dir}")
            logger.info(f"Video stills: {video_dir}")
        else:
            logger.info("=== DRY RUN - NO FILES COPIED ===")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during selection: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
