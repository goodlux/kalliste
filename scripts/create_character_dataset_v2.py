#!/usr/bin/env python3
"""
Create training dataset for a specific person (character LoRA).
- Copies ALL photographs for the person
- If video-count specified: Uses clustering-first approach for diversity
- If no video-count: Copies ALL video stills
- Creates properly structured directories with ratio prefixes
- Automatically syncs to remote server

Usage:
    python create_character_dataset_v2.py --person KyCoLux --video-count 5000
    python create_character_dataset_v2.py --person KyCoLux  # Copies ALL videos
    python create_character_dataset.py --person KyCoLux --video-count 3000 --no-sync --dry-run
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil
import subprocess
import math

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

def get_all_videos_for_person(db: MilvusDB, person_name: str, min_technical_score: float = 0.45) -> List[Dict[str, Any]]:
    """Get ALL video stills for a person (no clustering)."""
    
    filter_expr = f'person_name == "{person_name}" && source_type == "video"'
    if min_technical_score > 0:
        filter_expr += f' && nima_score_technical >= {min_technical_score}'
    
    output_fields = [
        "id", "image_file_path", "nima_score_technical", "nima_score_aesthetic", 
        "nima_score_calc_average", "person_name", "photoshoot"
    ]
    
    logger.info(f"Querying for ALL video stills of {person_name}")
    
    results = db.query(
        filter_expr=filter_expr,
        output_fields=output_fields
    )
    
    logger.info(f"Found {len(results)} video stills for {person_name}")
    return results

def get_best_video_stills_for_person(db: MilvusDB, person_name: str, count: int, 
                                    min_technical_score: float = 0.45) -> List[Dict[str, Any]]:
    """Get the best video stills using clustering-first approach for maximum diversity."""
    
    filter_expr = f'person_name == "{person_name}" && source_type == "video"'
    if min_technical_score > 0:
        filter_expr += f' && nima_score_technical >= {min_technical_score}'
    
    logger.info(f"Using clustering-first approach for {person_name} (min technical score: {min_technical_score})")
    
    # Get ALL video stills with embeddings using query_iterator
    all_videos = get_all_videos_with_iterator(db, filter_expr)
    
    if not all_videos:
        raise ValueError("No videos found with embeddings")
    
    logger.info(f"Retrieved {len(all_videos)} videos with embeddings")
    
    # Cluster and select diverse videos
    return cluster_and_select_diverse_videos(all_videos, count)

def get_all_videos_with_iterator(db: MilvusDB, filter_expr: str) -> List[Dict[str, Any]]:
    """Get ALL videos with embeddings using Milvus query_iterator."""
    
    from pymilvus import Collection, connections
    
    # Connect to Milvus
    connections.connect(uri="http://localhost:19530")
    collection = Collection(db.collection_name)
    
    output_fields = [
        "id", "image_file_path", "nima_score_technical", "nima_score_aesthetic", 
        "nima_score_calc_average", "person_name", "photoshoot", "openclip_vector"
    ]
    
    logger.info("Getting ALL videos with embeddings using query_iterator")
    
    # Create query iterator with batch processing
    batch_size = 1000
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr=filter_expr,
        output_fields=output_fields
    )
    
    all_results = []
    batch_count = 0
    
    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
                
            batch_count += 1
            batch_with_embeddings = 0
            
            # Filter for records that have embeddings
            for result in batch_results:
                embedding = result.get('openclip_vector')
                if embedding and len(embedding) > 0:
                    all_results.append(result)
                    batch_with_embeddings += 1
            
            logger.info(f"Batch {batch_count}: {len(batch_results)} records, {batch_with_embeddings} with embeddings")
            
            # Progress update every 10 batches
            if batch_count % 10 == 0:
                logger.info(f"Progress: {len(all_results)} total videos collected so far...")
    
    finally:
        iterator.close()
        connections.disconnect("default")
    
    logger.info(f"Completed: collected {len(all_results)} videos with embeddings in {batch_count} batches")
    return all_results

def cluster_and_select_diverse_videos(video_results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Cluster ALL video frames for diversity and select best technical score from each cluster."""
    
    # Import required libraries
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    if len(video_results) <= target_count:
        logger.info(f"Only {len(video_results)} videos available, returning all")
        return video_results
    
    # Extract embeddings
    embeddings = []
    valid_videos = []
    
    for result in video_results:
        embedding = result.get('openclip_vector')
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
            valid_videos.append(result)
    
    if len(embeddings) < target_count:
        logger.warning(f"Only {len(embeddings)} videos have embeddings, returning all")
        return valid_videos
    
    logger.info(f"Clustering {len(embeddings)} embeddings into {target_count} clusters")
    
    # Convert to numpy array and normalize
    embeddings_array = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_array)
    
    # Perform K-means clustering
    logger.info("Running K-means clustering...")
    kmeans = KMeans(n_clusters=target_count, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(embeddings_normalized)
    
    # Select best video from each cluster based on technical score
    selected_videos = []
    cluster_stats = []
    
    logger.info("Selecting best video from each cluster...")
    
    for cluster_id in range(target_count):
        # Get videos in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {cluster_id} is empty")
            continue
            
        # Get the actual video results for this cluster
        cluster_videos = [valid_videos[i] for i in cluster_indices]
        
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
    
    # Log cluster statistics
    if cluster_stats:
        cluster_sizes = [s['size'] for s in cluster_stats]
        tech_scores = [s['best_technical'] for s in cluster_stats]
        
        logger.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        logger.info(f"Selected technical scores: {min(tech_scores):.3f} - {max(tech_scores):.3f}, avg={sum(tech_scores)/len(tech_scores):.3f}")
    
    return selected_videos

def calculate_ratio(photo_count: int, video_count: int) -> tuple[int, int]:
    """Calculate the simplified ratio between photo and video counts."""
    if photo_count == 0 or video_count == 0:
        return (1 if photo_count > 0 else 0, 1 if video_count > 0 else 0)
    
    # Calculate GCD to simplify ratio
    gcd = math.gcd(photo_count, video_count)
    photo_ratio = photo_count // gcd
    video_ratio = video_count // gcd
    
    # If ratios are too large, simplify further
    if photo_ratio > 10 or video_ratio > 10:
        # Normalize to smaller numbers
        total = photo_ratio + video_ratio
        photo_ratio = round((photo_ratio / total) * 10)
        video_ratio = round((video_ratio / total) * 10)
        
        # Ensure at least 1 for each if they exist
        if photo_ratio == 0 and photo_count > 0:
            photo_ratio = 1
        if video_ratio == 0 and video_count > 0:
            video_ratio = 1
    
    return photo_ratio, video_ratio

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

def sync_to_remote(local_dir: str, person_name: str, dry_run: bool = False):
    """Sync the created dataset to remote server using rsync."""
    
    remote_dest = f"lux@clxbox:/mnt/sda2/lora_source/"
    
    # Build rsync command
    cmd = [
        "rsync", "-avz", "--delete",
        local_dir,  # Source (PersonName directory)
        remote_dest  # Destination
    ]
    
    if dry_run:
        cmd.insert(1, "-n")  # Add dry-run flag after rsync
        logger.info(f"DRY RUN: Would sync to remote with command: {' '.join(cmd)}")
    else:
        logger.info(f"Syncing to remote server: {remote_dest}{person_name}")
        logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Remote sync completed successfully")
            if result.stdout:
                logger.debug(f"Rsync output:\n{result.stdout}")
        else:
            logger.error(f"Rsync failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Failed to sync to remote: {e}")

def create_dataset_summary(person_name: str, photo_results: List[Dict], video_results: List[Dict], 
                         output_base_dir: str, images_stats: Dict, 
                         min_technical_score: float = 0.45, 
                         clustering_used: bool = True,
                         dry_run: bool = False):
    """Create a summary file for the dataset."""
    
    if dry_run:
        return
    
    summary_path = Path(output_base_dir) / f"{person_name}_dataset_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"Character Dataset: {person_name}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if clustering_used:
            f.write(f"Selection method: Clustering-First (ALL videos clustered for maximum diversity)\n")
        else:
            f.write(f"Selection method: ALL videos included (no clustering)\n")
            
        f.write(f"Min technical score threshold: {min_technical_score}\n")
        f.write("="*50 + "\n\n")
        
        # Photo summary
        f.write(f"PHOTOGRAPHS: {len(photo_results)} images (all available)\n")
        if photo_results:
            photo_nima_scores = [r.get('nima_score_technical', 0) for r in photo_results]
            f.write(f"  NIMA Technical Score Range: {min(photo_nima_scores):.3f} - {max(photo_nima_scores):.3f}\n")
            f.write(f"  Average NIMA Technical: {sum(photo_nima_scores)/len(photo_nima_scores):.3f}\n")
        f.write(f"  Files copied: {images_stats['photos_copied']} images, {images_stats['photo_sidecars']} sidecars\n\n")
        
        # Video summary
        if clustering_used:
            f.write(f"VIDEO STILLS: {len(video_results)} images (clustered ALL videos into {len(video_results)} diverse groups, best quality from each)\n")
        else:
            f.write(f"VIDEO STILLS: {len(video_results)} images (ALL videos meeting quality threshold)\n")
            
        if video_results:
            video_nima_scores = [r.get('nima_score_technical', 0) for r in video_results]
            f.write(f"  NIMA Technical Score Range: {min(video_nima_scores):.3f} - {max(video_nima_scores):.3f}\n")
            f.write(f"  Average NIMA Technical: {sum(video_nima_scores)/len(video_nima_scores):.3f}\n")
        f.write(f"  Files copied: {images_stats['videos_copied']} images, {images_stats['video_sidecars']} sidecars\n\n")
        
        # Totals
        total_images = images_stats['photos_copied'] + images_stats['videos_copied']
        total_sidecars = images_stats['photo_sidecars'] + images_stats['video_sidecars']
        f.write(f"TOTAL: {total_images} images, {total_sidecars} sidecar files\n")
        
        # Ratio information
        f.write(f"\nRatio: {images_stats['photo_ratio']} photos : {images_stats['video_ratio']} videos\n")
    
    logger.info(f"Dataset summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Create character training dataset with automatic remote sync")
    parser.add_argument("--person", required=True, help="Person identifier (e.g., KyCoLux)")
    parser.add_argument("--video-count", type=int, default=None, 
                        help="Number of video stills to select (if not specified, copies ALL videos)")
    parser.add_argument("--min-technical", type=float, default=0.45,
                        help="Minimum NIMA technical score threshold for video stills")
    parser.add_argument("--output-base", default="/Volumes/m01/kalliste_data/datasets",
                        help="Base output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without actually copying")
    parser.add_argument("--no-sync", action="store_true",
                        help="Don't sync to remote server after creating dataset")
    
    args = parser.parse_args()
    
    # Setup paths - now with new structure
    person_dir = os.path.join(args.output_base, args.person)
    images_dir = os.path.join(person_dir, "images")
    lora_dir = os.path.join(person_dir, "lora")
    logs_dir = os.path.join(person_dir, "logs")
    
    logger.info(f"=== CREATING CHARACTER DATASET FOR {args.person} ===")
    logger.info(f"Base directory: {person_dir}")
    
    if args.video_count:
        logger.info(f"Target video stills: {args.video_count} (using clustering)")
    else:
        logger.info("Target video stills: ALL (no clustering)")
        
    logger.info(f"Min technical score: {args.min_technical}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
    
    # Initialize database
    db = MilvusDB()
    
    try:
        # Get all photographs
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Getting all photographs")
        logger.info("="*50)
        
        photo_results = get_all_photos_for_person(db, args.person)
        
        # Get video stills - either clustered or all
        logger.info("\n" + "="*50)
        if args.video_count:
            logger.info("STEP 2: Clustering ALL videos for diversity")
            video_results = get_best_video_stills_for_person(
                db, args.person, args.video_count, 
                min_technical_score=args.min_technical
            )
            clustering_used = True
        else:
            logger.info("STEP 2: Getting ALL video stills (no clustering)")
            video_results = get_all_videos_for_person(
                db, args.person,
                min_technical_score=args.min_technical
            )
            clustering_used = False
        logger.info("="*50)
        
        # Calculate ratio
        photo_count = len(photo_results)
        video_count = len(video_results)
        photo_ratio, video_ratio = calculate_ratio(photo_count, video_count)
        
        logger.info(f"\nImage counts: {photo_count} photos, {video_count} videos")
        logger.info(f"Ratio: {photo_ratio}:{video_ratio}")
        
        # Setup directory names with ratios
        photo_dir = os.path.join(images_dir, f"{photo_ratio}_photo")
        video_dir = os.path.join(images_dir, f"{video_ratio}_video")
        
        # Create all directories (including empty lora and logs)
        if not args.dry_run:
            os.makedirs(photo_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(lora_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            logger.info("Created directory structure")
        
        # Copy photographs
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Copying photographs")
        logger.info("="*50)
        
        photos_copied, photo_sidecars = copy_images_and_sidecars(
            photo_results, photo_dir, "photographs", args.dry_run
        )
        
        # Copy video stills
        logger.info("\n" + "="*50)
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
            'video_sidecars': video_sidecars,
            'photo_ratio': photo_ratio,
            'video_ratio': video_ratio
        }
        
        create_dataset_summary(args.person, photo_results, video_results, 
                             person_dir, images_stats, 
                             args.min_technical, clustering_used, args.dry_run)
        
        # Sync to remote server
        if not args.no_sync and not args.dry_run:
            logger.info("\n" + "="*50)
            logger.info("STEP 5: Syncing to remote server")
            logger.info("="*50)
            sync_to_remote(person_dir, args.person, args.dry_run)
        elif args.no_sync:
            logger.info("\nSkipping remote sync (--no-sync specified)")
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("DATASET CREATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Person: {args.person}")
        logger.info(f"Photographs: {photos_copied} images, {photo_sidecars} sidecars")
        logger.info(f"Video stills: {videos_copied} images, {video_sidecars} sidecars")
        logger.info(f"Total: {photos_copied + videos_copied} images, {photo_sidecars + video_sidecars} sidecars")
        logger.info(f"Directory structure:")
        logger.info(f"  {person_dir}/")
        logger.info(f"  ├── images/")
        logger.info(f"  │   ├── {photo_ratio}_photo/")
        logger.info(f"  │   └── {video_ratio}_video/")
        logger.info(f"  ├── lora/")
        logger.info(f"  └── logs/")
        
        if not args.dry_run:
            logger.info(f"\nDataset saved to: {person_dir}")
            if not args.no_sync:
                logger.info(f"Remote location: lux@clxbox:/mnt/sda2/lora_source/{args.person}")
        else:
            logger.info("\nDRY RUN COMPLETE - No files were copied")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
