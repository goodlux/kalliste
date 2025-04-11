#!/usr/bin/env python3
"""
Script to select images from Milvus database using a query expression 
and export them to a dataset folder.

Examples:
    # Select acceptable images from a specific person:
    python select_milvus_images.py --query "person_name == 'ArMcLux' && nima_assessment_overall == 'acceptable'" --dataset_name "armclux_acceptable"

    # Select images with a limit:
    python select_milvus_images.py --query "person_name == 'ArMcLux' && nima_assessment_overall == 'acceptable'" --dataset_name "armclux_acceptable" --limit 50
    
    # Select images with good rating:
    python select_milvus_images.py --query "lr_rating >= 3" --dataset_name "high_rated_images"

    # Select images by photoshoot:
    python select_milvus_images.py --query "photoshoot == 'Studio2024'" --dataset_name "studio_photoshoot"
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB
from kalliste.config import KALLISTE_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Select and export images from Milvus database")
    parser.add_argument("--query", required=True, help="Milvus filter expression")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset folder to create")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to export (default: no limit)")
    parser.add_argument("--output_dir", default=None, 
                        help=f"Base output directory (default: {os.path.join(KALLISTE_DATA_DIR, 'datasets')})")
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(KALLISTE_DATA_DIR, "datasets")
    
    # Create destination folder path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_folder = os.path.join(args.output_dir, f"{args.dataset_name}_{timestamp}")
    
    # Initialize database connection
    db = MilvusDB()
    
    try:
        # Execute the query
        logger.info(f"Querying Milvus with expression: {args.query}")
        limit_msg = f" (limit: {args.limit})" if args.limit else " (no limit)"
        logger.info(f"Running query{limit_msg}")
        
        results = db.query(
            filter_expr=args.query,
            limit=args.limit,
            output_fields=["id", "image_file_path", "person_name", "photoshoot", 
                           "nima_score_calc_average", "lr_rating", "nima_assessment_overall"]
        )
        
        if not results:
            logger.error("Query returned no results.")
            return 1
            
        logger.info(f"Found {len(results)} matching images")
        
        # Export images to destination folder
        logger.info(f"Exporting images to {dataset_folder}")
        copied_files = db.export_images(results, dataset_folder)
        
        if not copied_files:
            logger.error("Failed to copy any files.")
            return 1
            
        logger.info(f"Successfully copied {len(copied_files)} images to {dataset_folder}")
        
        # Create a summary file
        summary_path = os.path.join(dataset_folder, "dataset_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Query: {args.query}\n")
            f.write(f"Total images: {len(copied_files)}\n\n")
            
            f.write("Image details:\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {os.path.basename(result['image_file_path'])}\n")
                if "person_name" in result:
                    f.write(f"   Person: {result.get('person_name', 'Unknown')}\n")
                if "photoshoot" in result:
                    f.write(f"   Photoshoot: {result.get('photoshoot', 'Unknown')}\n")
                if "nima_score_calc_average" in result:
                    f.write(f"   NIMA Score: {result.get('nima_score_calc_average', 0):.2f}\n")
                if "lr_rating" in result:
                    f.write(f"   LR Rating: {result.get('lr_rating', 0)}\n")
                f.write("\n")
        
        logger.info(f"Created summary file: {summary_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())