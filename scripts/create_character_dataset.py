#!/usr/bin/env python3
"""
Create training dataset for a specific person (character LoRA).
- Copies ALL photographs for the person
- Selects BEST video stills using clustering for diversity + NIMA technical score for quality
- Copies both image files and sidecar .txt files

Usage:
    python create_character_dataset.py --person NaGiLux --video-count 5000
    python create_character_dataset.py --person NaGiLux --video-count 5000 --no-clustering
    python create_character_dataset.py --person NaGiLux --video-count 3000 --min-technical 0.50 --dry-run
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any
import shutil

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_photos_for_person(db: MilvusDB, person_name: str) -> List[Dict[str, Any]]:
    """Get all photographs for a specific person."""
    
    filter_expr = f'person_name == "{person_name}" && source_type == "photograph"'
    
    output_fields = [
        "id", "image_file_path", "nima_score_technical", "nima_score_aesthetic", 
        "nima_score_calc_average", "person_name", "photoshoot"
    ]
    
    logger.info(f"Querying for all photographs of {person_name}")
    
    results = db.query(
        filter_expr=filter_expr,
        output_fields=output_fields
    )
    
    logger.info(f"Found {len(results)} photographs for {person_name}")
    return results

def get_best_video_stills_for_person(db: MilvusDB, person_name: str, count: int, 
                                    use_clustering: bool = True, min_technical_score: float = 0.45) -> List[Dict[str, Any]]:
    """Get the best video stills for a specific person.
    
    Uses a two-phase approach to avoid Milvus query size limits:
    1. Get top candidates by technical score (without embeddings)
    2. Get embeddings for selected candidates and cluster
    """
    
    filter_expr = f'person_name == "{person_name}" && source_type == "video"'
    if min_technical_score > 0:
        filter_expr += f' && nima_score_technical >= {min_technical_score}'
    
    logger.info(f"Selecting video stills for {person_name} (min technical score: {min_technical_score})")
    
    if not use_clustering:
        # Simple approach: just take top by technical score
        logger.info("Using simple technical score ranking (no clustering)")
        
        output_fields = [
            "id", "image_file_path", "nima_score_technical", "nima_score_aesthetic", 
            "nima_score_calc_average", "person_name", "photoshoot"
        ]
        
        # Get all results and sort by technical score
        all_results = db.query(
            filter_expr=filter_expr,
            output_fields=output_fields
        )
        
        logger.info(f"Found {len(all_results)} video stills above threshold")
        
        if not all_results:
            return []
        
        # Sort by technical score and take top count
        sorted_results = sorted(
            all_results,
            key=lambda x: x.get('nima_score_technical', 0.0),
            reverse=True
        )
        
        best_results = sorted_results[:count]
        
        if best_results:
            tech_scores = [r.get('nima_score_technical', 0.0) for r in best_results]
            logger.info(f"Selected {len(best_results)} videos by technical score")
            logger.info(f"NIMA technical score range: {min(tech_scores):.3f} - {max(tech_scores):.3f}")
        
        return best_results
    
    else:
        # Clustering approach - two-phase to avoid size limits
        logger.info("Using clustering for diversity")
        
        # Phase 1: Get more candidates than we need (without embeddings)
        # Use 3x the target count to have good candidates for clustering
        candidate_count = min(count * 3, 50000)  # Cap at 50k to avoid other issues
        
        logger.info(f"Phase 1: Getting top {candidate_count} candidates by technical score")
        
        output_fields_phase1 = [
            "id", "image_file_path", "nima_score_technical", "nima_score_aesthetic", 
            "nima_score_calc_average", "person_name", "photoshoot"
        ]
        
        # Get all candidates and sort by technical score
        all_candidates = db.query(
            filter_expr=filter_expr,
            output_fields=output_fields_phase1
        )
        
        logger.info(f"Found {len(all_candidates)} total video stills above threshold")
        
        if not all_candidates:
            return []
        
        # Sort by technical score and take top candidates
        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: x.get('nima_score_technical', 0.0),
            reverse=True
        )
        
        top_candidates = sorted_candidates[:candidate_count]
        logger.info(f"Selected top {len(top_candidates)} candidates for clustering")
        
        # Phase 2: Get embeddings for selected candidates in batches
        logger.info(f"Phase 2: Getting embeddings and clustering into {count} groups")
        
        candidates_with_embeddings = get_embeddings_for_candidates(db, top_candidates)
        
        if not candidates_with_embeddings:
            logger.warning("No candidates have embeddings, falling back to simple selection")
            return top_candidates[:count]
        
        # Cluster and select diverse videos
        return cluster_and_select_diverse_videos(candidates_with_embeddings, count)

def get_embeddings_for_candidates(db: MilvusDB, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get embeddings for a list of candidate videos in batches."""
    
    batch_size = 1000  # Process in smaller batches
    candidates_with_embeddings = []
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_ids = [str(c['id']) for c in batch]
        
        logger.info(f"Getting embeddings for batch {i//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}")
        
        try:
            # Query by ID to get embeddings
            id_filter = f"id in {batch_ids}"
            
            embedding_results = db.query(
                filter_expr=id_filter,
                output_fields=["id", "openclip_vector"]
            )
            
            # Merge embedding data with candidate data
            embedding_dict = {r['id']: r.get('openclip_vector') for r in embedding_results}
            
            for candidate in batch:
                candidate_id = candidate['id']
                embedding = embedding_dict.get(candidate_id)
                
                if embedding and len(embedding) > 0:
                    candidate['openclip_vector'] = embedding
                    candidates_with_embeddings.append(candidate)
                else:
                    logger.debug(f"No embedding found for candidate {candidate_id}")
                    
        except Exception as e:
            logger.error(f"Error getting embeddings for batch: {e}")
            # Skip this batch and continue
            continue
    
    logger.info(f"Successfully got embeddings for {len(candidates_with_embeddings)}/{len(candidates)} candidates")
    return candidates_with_embeddings

def cluster_and_select_diverse_videos(video_results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Cluster video frames for diversity and select best technical score from each cluster."""
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except ImportError:
        logger.error("sklearn required for clustering. Install with: pip install scikit-learn")
        logger.info("Falling back to simple technical score selection")
        # Fallback to simple sorting
        sorted_results = sorted(
            video_results,
            key=lambda x: x.get('nima_score_technical', 0.0),
            reverse=True
        )
        return sorted_results[:target_count]
    
    if len(video_results) <= target_count:
        logger.info(f"Only {len(video_results)} videos available, returning all")
        return video_results
    
    # Extract embeddings
    embeddings = []
    valid_indices = []
    
    for i, result in enumerate(video_results):
        embedding = result.get('openclip_vector')
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
            valid_indices.append(i)
        else:
            logger.warning(f"Skipping video {i} - missing embedding")
    
    if len(embeddings) < target_count:
        logger.warning(f"Only {len(embeddings)} videos have embeddings")
        return [video_results[i] for i in valid_indices]
    
    logger.info(f"Clustering {len(embeddings)} embeddings into {target_count} clusters")
    
    # Convert to numpy array and normalize
    embeddings_array = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_array)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=target_count, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_normalized)
    
    # Select best video from each cluster based on technical score
    selected_videos = []
    cluster_stats = []
    
    for cluster_id in range(target_count):
        # Get videos in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Get the actual video results for this cluster
        cluster_videos = [video_results[valid_indices[i]] for i in cluster_indices]
        
        # Sort by technical score and take the best
        cluster_videos.sort(key=lambda x: x.get('nima_score_technical', 0.0), reverse=True)
        best_video = cluster_videos[0]
        selected_videos.append(best_video)
        
        # Track statistics
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': len(cluster_videos),
            'best_technical': best_video.get('nima_score_technical', 0.0),
            'avg_technical': sum(v.get('nima_score_technical', 0.0) for v in cluster_videos) / len(cluster_videos)
        })
    
    logger.info(f"Successfully selected {len(selected_videos)} diverse videos from {target_count} clusters")
    
    # Log some cluster statistics
    if cluster_stats:
        cluster_sizes = [s['size'] for s in cluster_stats]
        logger.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        tech_scores = [s['best_technical'] for s in cluster_stats]
        logger.info(f"Selected technical scores: {min(tech_scores):.3f} - {max(tech_scores):.3f}")
    
    return selected_videos

def copy_images_and_sidecars(results: List[Dict[str, Any]], destination_dir: str, 
                           source_type: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Copy image files and their corresponding sidecar .txt files.
    
    Returns:
        Tuple of (images_copied, sidecars_copied)
    """
    
    if not results:
        logger.warning(f"No {source_type} results to copy")
        return 0, 0
    
    # Create destination directory
    dest_path = Path(destination_dir)
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created destination directory: {dest_path}")
    
    images_copied = 0
    sidecars_copied = 0
    missing_images = 0
    missing_sidecars = 0
    
    for i, result in enumerate(results, 1):
        image_path_str = result.get('image_file_path', '')
        if not image_path_str:
            logger.warning(f"Missing image_file_path in result {i}")
            continue
            
        # Source paths
        image_path = Path(image_path_str)
        sidecar_path = image_path.with_suffix('.txt')
        
        # Destination paths
        dest_image_path = dest_path / image_path.name
        dest_sidecar_path = dest_path / sidecar_path.name
        
        # Copy image file
        if image_path.exists():
            if not dry_run:
                try:
                    shutil.copy2(image_path, dest_image_path)
                    images_copied += 1
                except Exception as e:
                    logger.error(f"Error copying image {image_path}: {e}")
                    continue
            else:
                images_copied += 1  # Count for dry run
        else:
            logger.warning(f"Image file not found: {image_path}")
            missing_images += 1
            continue
        
        # Copy sidecar file
        if sidecar_path.exists():
            if not dry_run:
                try:
                    shutil.copy2(sidecar_path, dest_sidecar_path)
                    sidecars_copied += 1
                except Exception as e:
                    logger.error(f"Error copying sidecar {sidecar_path}: {e}")
            else:
                sidecars_copied += 1  # Count for dry run
        else:
            logger.warning(f"Sidecar file not found: {sidecar_path}")
            missing_sidecars += 1
        
        # Progress logging
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(results)} {source_type} files...")
    
    # Summary
    logger.info(f"=== {source_type.upper()} COPY SUMMARY ===")
    logger.info(f"Images copied: {images_copied}")
    logger.info(f"Sidecars copied: {sidecars_copied}")
    if missing_images > 0:
        logger.warning(f"Missing images: {missing_images}")
    if missing_sidecars > 0:
        logger.warning(f"Missing sidecars: {missing_sidecars}")
    
    return images_copied, sidecars_copied

def create_dataset_summary(person_name: str, photo_results: List[Dict], video_results: List[Dict], 
                         output_base_dir: str, images_stats: Dict, clustering_used: bool = True, 
                         min_technical_score: float = 0.45, dry_run: bool = False):
    """Create a summary file for the dataset."""
    
    if dry_run:
        return
    
    summary_path = Path(output_base_dir) / f"{person_name}_dataset_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"Character Dataset: {person_name}\\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Selection method: {'Clustering + Quality' if clustering_used else 'Quality Only'}\\n")
        f.write(f"Min technical score threshold: {min_technical_score}\\n")
        f.write("="*50 + "\\n\\n")
        
        # Photo summary
        f.write(f"PHOTOGRAPHS: {len(photo_results)} images\\n")
        if photo_results:
            photo_nima_scores = [r.get('nima_score_technical', 0) for r in photo_results]
            f.write(f"  NIMA Technical Score Range: {min(photo_nima_scores):.3f} - {max(photo_nima_scores):.3f}\\n")
            f.write(f"  Average NIMA Technical: {sum(photo_nima_scores)/len(photo_nima_scores):.3f}\\n")
        f.write(f"  Files copied: {images_stats['photos_copied']} images, {images_stats['photo_sidecars']} sidecars\\n\\n")
        
        # Video summary
        selection_method = "clustered for diversity" if clustering_used else "ranked by technical score"
        f.write(f"VIDEO STILLS: {len(video_results)} images ({selection_method})\\n")
        if video_results:
            video_nima_scores = [r.get('nima_score_technical', 0) for r in video_results]
            f.write(f"  NIMA Technical Score Range: {min(video_nima_scores):.3f} - {max(video_nima_scores):.3f}\\n")
            f.write(f"  Average NIMA Technical: {sum(video_nima_scores)/len(video_nima_scores):.3f}\\n")
        f.write(f"  Files copied: {images_stats['videos_copied']} images, {images_stats['video_sidecars']} sidecars\\n\\n")
        
        # Totals
        total_images = images_stats['photos_copied'] + images_stats['videos_copied']
        total_sidecars = images_stats['photo_sidecars'] + images_stats['video_sidecars']
        f.write(f"TOTAL: {total_images} images, {total_sidecars} sidecar files\\n")
    
    logger.info(f"Dataset summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Create character training dataset with clustering")
    parser.add_argument("--person", required=True, help="Person identifier (e.g., NaGiLux)")
    parser.add_argument("--video-count", type=int, default=5000, 
                        help="Number of video stills to select")
    parser.add_argument("--min-technical", type=float, default=0.45,
                        help="Minimum NIMA technical score threshold for video stills")
    parser.add_argument("--no-clustering", action="store_true",
                        help="Disable clustering - just take top videos by technical score")
    parser.add_argument("--output-base", default="/Volumes/m01/kalliste_data/datasets",
                        help="Base output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without actually copying")
    
    args = parser.parse_args()
    
    # Setup paths
    person_dir = os.path.join(args.output_base, args.person)
    photo_dir = os.path.join(person_dir, "photo")
    video_dir = os.path.join(person_dir, "video")
    
    use_clustering = not args.no_clustering
    
    logger.info(f"=== CREATING CHARACTER DATASET FOR {args.person} ===")
    logger.info(f"Photo destination: {photo_dir}")
    logger.info(f"Video destination: {video_dir}")
    logger.info(f"Target video stills: {args.video_count}")
    logger.info(f"Min technical score: {args.min_technical}")
    logger.info(f"Using clustering: {use_clustering}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
    
    # Initialize database
    db = MilvusDB()
    
    try:
        # Get all photographs
        logger.info("\\n" + "="*50)
        logger.info("STEP 1: Getting all photographs")
        logger.info("="*50)
        
        photo_results = get_all_photos_for_person(db, args.person)
        
        # Get best video stills
        logger.info("\\n" + "="*50)
        logger.info("STEP 2: Selecting best video stills")
        logger.info("="*50)
        
        video_results = get_best_video_stills_for_person(
            db, args.person, args.video_count, 
            use_clustering=use_clustering,
            min_technical_score=args.min_technical
        )
        
        # Copy photographs
        logger.info("\\n" + "="*50)
        logger.info("STEP 3: Copying photographs")
        logger.info("="*50)
        
        photos_copied, photo_sidecars = copy_images_and_sidecars(
            photo_results, photo_dir, "photographs", args.dry_run
        )
        
        # Copy video stills
        logger.info("\\n" + "="*50)
        logger.info("STEP 4: Copying video stills")
        logger.info("="*50)
        
        videos_copied, video_sidecars = copy_images_and_sidecars(
            video_results, video_dir, "video stills", args.dry_run
        )
        
        # Create summary
        images_stats = {
            'photos_copied': photos_copied,
            'photo_sidecars': photo_sidecars,
            'videos_copied': videos_copied,
            'video_sidecars': video_sidecars
        }
        
        create_dataset_summary(args.person, photo_results, video_results, 
                             person_dir, images_stats, use_clustering, 
                             args.min_technical, args.dry_run)
        
        # Final summary
        logger.info("\\n" + "="*50)
        logger.info("DATASET CREATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Person: {args.person}")
        logger.info(f"Photographs: {photos_copied} images, {photo_sidecars} sidecars")
        logger.info(f"Video stills: {videos_copied} images, {video_sidecars} sidecars")
        logger.info(f"Total: {photos_copied + videos_copied} images, {photo_sidecars + video_sidecars} sidecars")
        
        if not args.dry_run:
            logger.info(f"\\nDataset saved to: {person_dir}")
        else:
            logger.info("\\nDRY RUN COMPLETE - No files were copied")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
