#!/usr/bin/env python3
"""
Analysis script focused on NIMA technical scores for character dataset creation.

Usage:
    python analyze_character_data.py --person NaGiLux
    python analyze_character_data.py --person NaGiLux --min-technical 0.4
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Any
from collections import Counter

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_person_data(db: MilvusDB, person_name: str, min_technical_score: float = 0.0):
    """Analyze data available for a specific person."""
    
    logger.info(f"=== ANALYZING DATA FOR {person_name} ===")
    
    # Get all data for this person
    filter_expr = f'person_name == "{person_name}"'
    if min_technical_score > 0:
        filter_expr += f' && nima_score_technical >= {min_technical_score}'
        logger.info(f"Filtering for NIMA technical score >= {min_technical_score}")
    
    output_fields = [
        "id", "source_type", "nima_score_technical", "nima_score_aesthetic", 
        "nima_score_calc_average", "person_name", "photoshoot", "image_file_path"
    ]
    
    try:
        results = db.query(
            filter_expr=filter_expr,
            output_fields=output_fields
        )
        
        if not results:
            logger.error(f"No data found for {person_name}")
            return
            
        logger.info(f"Found {len(results)} total records for {person_name}")
        
        # Separate by source type
        photos = [r for r in results if r.get('source_type') == 'photograph']
        videos = [r for r in results if r.get('source_type') == 'video']
        
        print("\\n" + "="*60)
        print(f"DATA SUMMARY FOR {person_name}")
        print("="*60)
        print(f"Photographs: {len(photos)}")
        print(f"Video stills: {len(videos)}")
        print(f"Total: {len(results)}")
        
        # Analyze photographs
        if photos:
            print("\\n" + "-"*40)
            print("PHOTOGRAPH ANALYSIS")
            print("-"*40)
            analyze_image_set(photos, "photographs")
        
        # Analyze video stills
        if videos:
            print("\\n" + "-"*40)
            print("VIDEO STILLS ANALYSIS")
            print("-"*40)
            analyze_image_set(videos, "video stills")
            
            # Show top video stills for selection preview
            print("\\n" + "-"*40)
            print("TOP 10 VIDEO STILLS BY TECHNICAL SCORE")
            print("-"*40)
            top_videos = sorted(videos, key=lambda x: x.get('nima_score_technical', 0), reverse=True)[:10]
            for i, video in enumerate(top_videos, 1):
                tech_score = video.get('nima_score_technical', 0)
                aesthetic_score = video.get('nima_score_aesthetic', 0)
                filename = os.path.basename(video.get('image_file_path', ''))
                print(f"{i:2d}. {filename}")
                print(f"    Technical: {tech_score:.4f}, Aesthetic: {aesthetic_score:.4f}")
        
        # Photoshoot breakdown
        photoshoot_counts = Counter(r.get('photoshoot', 'Unknown') for r in results)
        print("\\n" + "-"*40)
        print("PHOTOSHOOT BREAKDOWN")
        print("-"*40)
        for shoot, count in photoshoot_counts.most_common():
            print(f"{shoot}: {count} images")
        
        # Recommendations
        print("\\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if photos:
            print(f"✓ Use ALL {len(photos)} photographs in training set")
        else:
            print("⚠ No photographs found")
            
        if videos:
            # Calculate how many good quality videos are available
            good_videos = [v for v in videos if v.get('nima_score_technical', 0) >= 0.45]
            excellent_videos = [v for v in videos if v.get('nima_score_technical', 0) >= 0.50]
            
            print(f"✓ {len(videos)} total video stills available")
            print(f"✓ {len(good_videos)} with technical score >= 0.45")
            print(f"✓ {len(excellent_videos)} with technical score >= 0.50")
            print(f"✓ Recommend selecting 5000 best video stills by technical score")
        else:
            print("⚠ No video stills found")
            
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")

def analyze_image_set(images: List[Dict], set_name: str):
    """Analyze a set of images (photos or videos)."""
    
    if not images:
        print(f"No {set_name} found")
        return
    
    # Extract scores
    tech_scores = [img.get('nima_score_technical', 0) for img in images if img.get('nima_score_technical') is not None]
    aesthetic_scores = [img.get('nima_score_aesthetic', 0) for img in images if img.get('nima_score_aesthetic') is not None]
    avg_scores = [img.get('nima_score_calc_average', 0) for img in images if img.get('nima_score_calc_average') is not None]
    
    print(f"Count: {len(images)}")
    
    if tech_scores:
        print(f"Technical NIMA scores:")
        print(f"  Range: {min(tech_scores):.4f} - {max(tech_scores):.4f}")
        print(f"  Average: {sum(tech_scores)/len(tech_scores):.4f}")
        
    if aesthetic_scores:
        print(f"Aesthetic NIMA scores:")
        print(f"  Range: {min(aesthetic_scores):.4f} - {max(aesthetic_scores):.4f}")
        print(f"  Average: {sum(aesthetic_scores)/len(aesthetic_scores):.4f}")
        
    if avg_scores:
        print(f"Combined NIMA scores:")
        print(f"  Range: {min(avg_scores):.4f} - {max(avg_scores):.4f}")
        print(f"  Average: {sum(avg_scores)/len(avg_scores):.4f}")
    
    # Quality buckets based on technical score
    if tech_scores:
        excellent = len([s for s in tech_scores if s >= 0.55])
        very_good = len([s for s in tech_scores if 0.50 <= s < 0.55])
        good = len([s for s in tech_scores if 0.45 <= s < 0.50])
        acceptable = len([s for s in tech_scores if 0.40 <= s < 0.45])
        poor = len([s for s in tech_scores if s < 0.40])
        
        total = len(tech_scores)
        print(f"Quality distribution (by technical score):")
        print(f"  Excellent (0.55+): {excellent} ({excellent/total*100:.1f}%)")
        print(f"  Very Good (0.50-0.55): {very_good} ({very_good/total*100:.1f}%)")
        print(f"  Good (0.45-0.50): {good} ({good/total*100:.1f}%)")
        print(f"  Acceptable (0.40-0.45): {acceptable} ({acceptable/total*100:.1f}%)")
        print(f"  Poor (<0.40): {poor} ({poor/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze character data for dataset creation")
    parser.add_argument("--person", required=True, help="Person identifier (e.g., NaGiLux)")
    parser.add_argument("--min-technical", type=float, default=0.0,
                        help="Minimum technical NIMA score to include")
    
    args = parser.parse_args()
    
    # Initialize database connection
    logger.info("Connecting to Milvus database...")
    db = MilvusDB()
    
    try:
        analyze_person_data(db, args.person, args.min_technical)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
