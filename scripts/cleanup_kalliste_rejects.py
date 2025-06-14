#!/usr/bin/env python3
"""
Clean up images rejected by Kalliste logic from Milvus database and filesystem.
- Finds all records where (nima_assessment_technical != "high_quality") AND (nima_assessment_overall != "acceptable")
- Moves image files to images_delete folder
- Removes records from Milvus database

Usage:
    python cleanup_kalliste_rejects.py --dry-run  # See what would be deleted
    python cleanup_kalliste_rejects.py            # Actually delete
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any
import shutil

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_kalliste_rejected_images(db: MilvusDB) -> List[Dict[str, Any]]:
    """Find all NaGiLux images that should be rejected by Kalliste logic."""
    
    # Reject if: (technical != "high_quality") AND (overall != "acceptable") AND person is NaGiLux
    filter_expr = '(nima_assessment_technical != "high_quality") and (nima_assessment_overall != "acceptable") and (person_name == "NaGiLux")'
    
    output_fields = [
        "id", "image_file_path", 
        "nima_assessment_technical", "nima_assessment_overall",
        "nima_score_technical", "nima_score_aesthetic",
        "person_name", "source_type"
    ]
    
    logger.info("Querying for NaGiLux images rejected by Kalliste logic...")
    logger.info("Reject criteria: (technical != 'high_quality') AND (overall != 'acceptable') AND person == 'NaGiLux'")
    
    try:
        # Use query_iterator for large datasets
        from pymilvus import Collection, connections
        
        connections.connect(uri="http://localhost:19530")
        collection = Collection(db.collection_name)
        
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
                all_results.extend(batch_results)
                
                logger.info(f"Batch {batch_count}: {len(batch_results)} rejected images found")
                
                # Progress update every 10 batches
                if batch_count % 10 == 0:
                    logger.info(f"Progress: {len(all_results)} total rejected images collected...")
        
        finally:
            iterator.close()
            connections.disconnect("default")
        
        logger.info(f"Found {len(all_results)} NaGiLux images rejected by Kalliste logic")
        return all_results
        
    except Exception as e:
        logger.error(f"Error querying rejected images: {e}")
        return []

def move_rejected_files(rejected_images: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, int]:
    """Move rejected image files to delete folder."""
    
    base_images_dir = Path("/Volumes/m01/kalliste_data/images")
    base_delete_dir = Path("/Volumes/m01/kalliste_data/images_delete")
    
    stats = {
        'images_moved': 0,
        'sidecars_moved': 0,
        'missing_images': 0,
        'missing_sidecars': 0,
        'errors': 0
    }
    
    if not dry_run:
        base_delete_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created delete directory: {base_delete_dir}")
    
    for i, result in enumerate(rejected_images, 1):
        try:
            image_path_str = result.get('image_file_path', '')
            if not image_path_str:
                logger.warning(f"Missing image_file_path in result {i}")
                stats['errors'] += 1
                continue
            
            image_path = Path(image_path_str)
            sidecar_path = image_path.with_suffix('.txt')
            
            # Calculate relative path from base images directory
            try:
                rel_path = image_path.relative_to(base_images_dir)
            except ValueError:
                logger.error(f"Image path not under base images dir: {image_path}")
                stats['errors'] += 1
                continue
            
            # Create destination paths
            dest_image_path = base_delete_dir / rel_path
            dest_sidecar_path = dest_image_path.with_suffix('.txt')
            
            # Create destination directory
            if not dry_run:
                dest_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move image file
            if image_path.exists():
                if not dry_run:
                    try:
                        shutil.move(str(image_path), str(dest_image_path))
                        stats['images_moved'] += 1
                    except Exception as e:
                        logger.error(f"Error moving image {image_path}: {e}")
                        stats['errors'] += 1
                        continue
                else:
                    stats['images_moved'] += 1  # Count for dry run
            else:
                logger.warning(f"Image file not found: {image_path}")
                stats['missing_images'] += 1
                continue
            
            # Move sidecar file
            if sidecar_path.exists():
                if not dry_run:
                    try:
                        shutil.move(str(sidecar_path), str(dest_sidecar_path))
                        stats['sidecars_moved'] += 1
                    except Exception as e:
                        logger.error(f"Error moving sidecar {sidecar_path}: {e}")
                        stats['errors'] += 1
                else:
                    stats['sidecars_moved'] += 1  # Count for dry run
            else:
                stats['missing_sidecars'] += 1
            
            # Progress logging
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(rejected_images)} files...")
                
        except Exception as e:
            logger.error(f"Unexpected error processing result {i}: {e}")
            stats['errors'] += 1
    
    return stats

def delete_rejected_from_milvus(db: MilvusDB, rejected_images: List[Dict[str, Any]], dry_run: bool = False) -> int:
    """Delete rejected image records from Milvus."""
    
    if dry_run:
        logger.info(f"DRY RUN: Would delete {len(rejected_images)} records from Milvus")
        return len(rejected_images)
    
    # Extract IDs to delete
    ids_to_delete = [str(result['id']) for result in rejected_images]
    
    logger.info(f"Deleting {len(ids_to_delete)} records from Milvus...")
    
    try:
        from pymilvus import Collection, connections
        
        connections.connect(uri="http://localhost:19530")
        collection = Collection(db.collection_name)
        
        # Delete in batches to avoid timeout
        batch_size = 1000
        deleted_count = 0
        
        for i in range(0, len(ids_to_delete), batch_size):
            batch_ids = ids_to_delete[i:i + batch_size]
            
            # Create delete expression
            id_list = ",".join(batch_ids)
            delete_expr = f"id in [{id_list}]"
            
            collection.delete(delete_expr)
            deleted_count += len(batch_ids)
            
            logger.info(f"Deleted batch {i//batch_size + 1}: {len(batch_ids)} records")
        
        connections.disconnect("default")
        logger.info(f"Successfully deleted {deleted_count} records from Milvus")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting from Milvus: {e}")
        return 0

def print_summary_stats(rejected_images: List[Dict[str, Any]]):
    """Print summary statistics about rejected images."""
    
    if not rejected_images:
        logger.info("No rejected images found")
        return
    
    logger.info(f"\n{'='*50}")
    logger.info("NAGILUX REJECTED IMAGES SUMMARY")
    logger.info(f"{'='*50}")
    
    # Count by source type
    source_counts = {}
    person_counts = {}
    
    for result in rejected_images:
        source_type = result.get('source_type', 'unknown')
        person_name = result.get('person_name', 'unknown')
        
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        person_counts[person_name] = person_counts.get(person_name, 0) + 1
    
    logger.info(f"Total rejected images: {len(rejected_images):,}")
    logger.info(f"")
    logger.info("By source type:")
    for source_type, count in sorted(source_counts.items()):
        logger.info(f"  {source_type}: {count:,}")
    
    logger.info(f"")
    logger.info("Top 10 people with most rejected images:")
    for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {person}: {count:,}")
    
    # Technical score stats
    tech_scores = [r.get('nima_score_technical', 0) for r in rejected_images if r.get('nima_score_technical')]
    if tech_scores:
        logger.info(f"")
        logger.info(f"Technical score range: {min(tech_scores):.3f} - {max(tech_scores):.3f}")
        logger.info(f"Average technical score: {sum(tech_scores)/len(tech_scores):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Clean up NaGiLux images rejected by Kalliste logic from Milvus and filesystem")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show statistics, don't move or delete anything")
    
    args = parser.parse_args()
    
    logger.info(f"=== CLEANING UP NAGILUX REJECTED IMAGES ===")
    logger.info(f"Reject criteria: (technical != 'high_quality') AND (overall != 'acceptable') AND person == 'NaGiLux'")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be moved or deleted")
    elif args.stats_only:
        logger.info("STATS ONLY MODE - No files will be moved or deleted")
    
    # Initialize database
    db = MilvusDB()
    
    try:
        # Find all rejected images
        logger.info("\\n" + "="*50)
        logger.info("STEP 1: Finding NaGiLux images rejected by Kalliste logic")
        logger.info("="*50)
        
        rejected_images = find_kalliste_rejected_images(db)
        
        if not rejected_images:
            logger.info("No rejected images found. Cleanup complete!")
            return 0
        
        # Print summary statistics
        print_summary_stats(rejected_images)
        
        if args.stats_only:
            logger.info("\\nSTATS ONLY MODE - Stopping here")
            return 0
        
        # Move files to delete folder
        logger.info("\\n" + "="*50)
        logger.info("STEP 2: Moving image files to delete folder")
        logger.info("="*50)
        
        file_stats = move_rejected_files(rejected_images, args.dry_run)
        
        logger.info(f"\\n=== FILE MOVE SUMMARY ===")
        logger.info(f"Images moved: {file_stats['images_moved']:,}")
        logger.info(f"Sidecars moved: {file_stats['sidecars_moved']:,}")
        logger.info(f"Missing images: {file_stats['missing_images']:,}")
        logger.info(f"Missing sidecars: {file_stats['missing_sidecars']:,}")
        logger.info(f"Errors: {file_stats['errors']:,}")
        
        # Delete from Milvus
        logger.info("\\n" + "="*50)
        logger.info("STEP 3: Deleting records from Milvus")
        logger.info("="*50)
        
        deleted_count = delete_rejected_from_milvus(db, rejected_images, args.dry_run)
        
        # Final summary
        logger.info("\\n" + "="*50)
        logger.info("CLEANUP COMPLETE")
        logger.info("="*50)
        logger.info(f"Rejected images found: {len(rejected_images):,}")
        logger.info(f"Files moved: {file_stats['images_moved']:,} images, {file_stats['sidecars_moved']:,} sidecars")
        logger.info(f"Records deleted from Milvus: {deleted_count:,}")
        logger.info(f"Remaining NaGiLux in database: ~{276982 - len(rejected_images):,} accepted images")
        
        if args.dry_run:
            logger.info("\\nDRY RUN COMPLETE - No actual changes made")
        else:
            logger.info(f"\\nFiles moved to: /Volumes/m01/kalliste_data/images_delete/")
            logger.info("Your dataset is now much cleaner for clustering!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
