#!/usr/bin/env python3
"""
Analysis script to understand what's available in the Milvus database
for training set creation.

Usage:
    python analyze_milvus_content.py
    python analyze_milvus_content.py --min-nima 6.0
"""
import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Any
from collections import Counter, defaultdict
import json

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_database_content(db: MilvusDB, min_nima_score: float = 0.0):
    """Analyze the content available in the Milvus database."""
    
    # Get basic statistics for all data
    logger.info("=== ANALYZING MILVUS DATABASE CONTENT ===")
    
    # Build filter for minimum NIMA score if specified
    filter_expr = None
    if min_nima_score > 0:
        filter_expr = f"nima_score_calc_average >= {min_nima_score}"
        logger.info(f"Filtering for NIMA score >= {min_nima_score}")
    
    # Get all records with key fields
    output_fields = [
        "id", "source_type", "nima_score_calc_average", "lr_rating", 
        "person_name", "photoshoot", "photoshoot_date", "image_file_path"
    ]
    
    try:
        results = db.query(
            filter_expr=filter_expr or "id > 0",  # Get all if no filter
            output_fields=output_fields
        )
        
        if not results:
            logger.error("No data found in database")
            return
            
        logger.info(f"Found {len(results)} total records")
        
        # Analyze by source type
        source_type_analysis = analyze_by_source_type(results)
        print_source_type_analysis(source_type_analysis)
        
        # Analyze quality distribution
        quality_analysis = analyze_quality_distribution(results)
        print_quality_analysis(quality_analysis)
        
        # Analyze by person
        person_analysis = analyze_by_person(results)
        print_person_analysis(person_analysis)
        
        # Analyze by photoshoot
        photoshoot_analysis = analyze_by_photoshoot(results)
        print_photoshoot_analysis(photoshoot_analysis)
        
        # Recommendations
        print_recommendations(source_type_analysis, quality_analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing database: {e}")

def analyze_by_source_type(results: List[Dict]) -> Dict:
    """Analyze images by source type."""
    source_stats = defaultdict(lambda: {
        'count': 0,
        'nima_scores': [],
        'lr_ratings': [],
        'persons': set(),
        'photoshoots': set()
    })
    
    for result in results:
        source_type = result.get('source_type', 'unknown')
        source_stats[source_type]['count'] += 1
        
        if result.get('nima_score_calc_average') is not None:
            source_stats[source_type]['nima_scores'].append(result['nima_score_calc_average'])
            
        if result.get('lr_rating') is not None:
            source_stats[source_type]['lr_ratings'].append(result['lr_rating'])
            
        if result.get('person_name'):
            source_stats[source_type]['persons'].add(result['person_name'])
            
        if result.get('photoshoot'):
            source_stats[source_type]['photoshoots'].add(result['photoshoot'])
    
    return dict(source_stats)

def analyze_quality_distribution(results: List[Dict]) -> Dict:
    """Analyze quality score distributions."""
    nima_scores = [r.get('nima_score_calc_average') for r in results if r.get('nima_score_calc_average') is not None]
    lr_ratings = [r.get('lr_rating') for r in results if r.get('lr_rating') is not None]
    
    # NIMA score buckets
    nima_buckets = {
        'excellent (8.0+)': len([s for s in nima_scores if s >= 8.0]),
        'very_good (7.0-8.0)': len([s for s in nima_scores if 7.0 <= s < 8.0]),
        'good (6.0-7.0)': len([s for s in nima_scores if 6.0 <= s < 7.0]),
        'acceptable (5.0-6.0)': len([s for s in nima_scores if 5.0 <= s < 6.0]),
        'poor (<5.0)': len([s for s in nima_scores if s < 5.0])
    }
    
    # LR rating distribution
    lr_distribution = Counter(lr_ratings)
    
    return {
        'nima_stats': {
            'count': len(nima_scores),
            'min': min(nima_scores) if nima_scores else 0,
            'max': max(nima_scores) if nima_scores else 0,
            'average': sum(nima_scores) / len(nima_scores) if nima_scores else 0,
            'buckets': nima_buckets
        },
        'lr_stats': {
            'count': len(lr_ratings),
            'distribution': dict(lr_distribution)
        }
    }

def analyze_by_person(results: List[Dict]) -> Dict:
    """Analyze images by person."""
    person_stats = defaultdict(lambda: {
        'count': 0,
        'source_types': Counter(),
        'avg_nima': 0,
        'nima_scores': []
    })
    
    for result in results:
        person = result.get('person_name', 'Unknown')
        person_stats[person]['count'] += 1
        
        source_type = result.get('source_type', 'unknown')
        person_stats[person]['source_types'][source_type] += 1
        
        if result.get('nima_score_calc_average') is not None:
            person_stats[person]['nima_scores'].append(result['nima_score_calc_average'])
    
    # Calculate averages
    for person, stats in person_stats.items():
        if stats['nima_scores']:
            stats['avg_nima'] = sum(stats['nima_scores']) / len(stats['nima_scores'])
    
    return dict(person_stats)

def analyze_by_photoshoot(results: List[Dict]) -> Dict:
    """Analyze images by photoshoot."""
    photoshoot_stats = defaultdict(lambda: {
        'count': 0,
        'source_types': Counter(),
        'persons': set(),
        'avg_nima': 0,
        'nima_scores': []
    })
    
    for result in results:
        photoshoot = result.get('photoshoot', 'Unknown')
        photoshoot_stats[photoshoot]['count'] += 1
        
        source_type = result.get('source_type', 'unknown')
        photoshoot_stats[photoshoot]['source_types'][source_type] += 1
        
        if result.get('person_name'):
            photoshoot_stats[photoshoot]['persons'].add(result['person_name'])
            
        if result.get('nima_score_calc_average') is not None:
            photoshoot_stats[photoshoot]['nima_scores'].append(result['nima_score_calc_average'])
    
    # Calculate averages
    for shoot, stats in photoshoot_stats.items():
        if stats['nima_scores']:
            stats['avg_nima'] = sum(stats['nima_scores']) / len(stats['nima_scores'])
    
    return dict(photoshoot_stats)

def print_source_type_analysis(source_stats: Dict):
    """Print source type analysis."""
    print("\\n" + "="*60)
    print("SOURCE TYPE ANALYSIS")
    print("="*60)
    
    for source_type, stats in sorted(source_stats.items()):
        print(f"\\n{source_type.upper()}: {stats['count']} images")
        
        if stats['nima_scores']:
            avg_nima = sum(stats['nima_scores']) / len(stats['nima_scores'])
            min_nima = min(stats['nima_scores'])
            max_nima = max(stats['nima_scores'])
            print(f"  NIMA scores: {avg_nima:.2f} avg ({min_nima:.2f} - {max_nima:.2f})")
        
        if stats['lr_ratings']:
            avg_lr = sum(stats['lr_ratings']) / len(stats['lr_ratings'])
            print(f"  LR ratings: {avg_lr:.1f} avg")
            
        print(f"  Unique persons: {len(stats['persons'])}")
        print(f"  Unique photoshoots: {len(stats['photoshoots'])}")

def print_quality_analysis(quality_stats: Dict):
    """Print quality analysis."""
    print("\\n" + "="*60)
    print("QUALITY DISTRIBUTION")
    print("="*60)
    
    nima_stats = quality_stats['nima_stats']
    print(f"\\nNIMA SCORES ({nima_stats['count']} images with scores):")
    print(f"  Average: {nima_stats['average']:.2f}")
    print(f"  Range: {nima_stats['min']:.2f} - {nima_stats['max']:.2f}")
    print("\\n  Quality Buckets:")
    for bucket, count in nima_stats['buckets'].items():
        percentage = (count / nima_stats['count'] * 100) if nima_stats['count'] > 0 else 0
        print(f"    {bucket}: {count} images ({percentage:.1f}%)")
    
    lr_stats = quality_stats['lr_stats']
    print(f"\\nLIGHTROOM RATINGS ({lr_stats['count']} images with ratings):")
    for rating in sorted(lr_stats['distribution'].keys()):
        count = lr_stats['distribution'][rating]
        percentage = (count / lr_stats['count'] * 100) if lr_stats['count'] > 0 else 0
        stars = "★" * int(rating) if rating > 0 else "No rating"
        print(f"    {rating}: {count} images ({percentage:.1f}%) {stars}")

def print_person_analysis(person_stats: Dict):
    """Print person analysis."""
    print("\\n" + "="*60)
    print("PERSON ANALYSIS (Top 10)")
    print("="*60)
    
    # Sort by count, take top 10
    sorted_persons = sorted(person_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    
    for person, stats in sorted_persons:
        print(f"\\n{person}: {stats['count']} images")
        print(f"  Avg NIMA: {stats['avg_nima']:.2f}")
        print(f"  Source types: {dict(stats['source_types'])}")

def print_photoshoot_analysis(photoshoot_stats: Dict):
    """Print photoshoot analysis."""
    print("\\n" + "="*60)
    print("PHOTOSHOOT ANALYSIS (Top 10)")
    print("="*60)
    
    # Sort by count, take top 10
    sorted_shoots = sorted(photoshoot_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
    
    for shoot, stats in sorted_shoots:
        print(f"\\n{shoot}: {stats['count']} images")
        print(f"  Avg NIMA: {stats['avg_nima']:.2f}")
        print(f"  Persons: {len(stats['persons'])}")
        print(f"  Source types: {dict(stats['source_types'])}")

def print_recommendations(source_stats: Dict, quality_stats: Dict):
    """Print recommendations for training set creation."""
    print("\\n" + "="*60)
    print("RECOMMENDATIONS FOR TRAINING SET CREATION")
    print("="*60)
    
    # Analyze what's available for each source type
    photo_count = source_stats.get('photograph', {}).get('count', 0)
    video_count = source_stats.get('video_still', {}).get('count', 0)
    
    print(f"\\nAvailable for training sets:")
    print(f"  Photographs: {photo_count}")
    print(f"  Video stills: {video_count}")
    
    # Quality recommendations
    nima_buckets = quality_stats['nima_stats']['buckets']
    good_plus = nima_buckets['excellent (8.0+)'] + nima_buckets['very_good (7.0-8.0)'] + nima_buckets['good (6.0-7.0)']
    
    print(f"\\nQuality recommendations:")
    print(f"  High quality (6.0+ NIMA): {good_plus} images available")
    print(f"  Consider minimum NIMA threshold of 5.5-6.0 for training")
    
    # Balance recommendations
    if photo_count > 0 and video_count > 0:
        ratio = min(photo_count, video_count) / max(photo_count, video_count)
        print(f"\\nBalance analysis:")
        print(f"  Photo/Video ratio: {ratio:.2f}")
        if ratio < 0.5:
            print(f"  Warning: Imbalanced dataset - consider equal sampling")
        
        # Suggest target counts
        suggested_target = min(5000, min(photo_count, video_count))
        print(f"  Suggested target per category: {suggested_target}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Milvus database content")
    parser.add_argument("--min-nima", type=float, default=0.0,
                        help="Minimum NIMA score to include in analysis")
    
    args = parser.parse_args()
    
    # Initialize database connection
    logger.info("Connecting to Milvus database...")
    db = MilvusDB()
    
    try:
        analyze_database_content(db, args.min_nima)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
