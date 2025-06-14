#!/usr/bin/env python3
"""
Get accurate GROUP BY counts for nima_assessment_overall values using query_iterator.
"""
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB
from pymilvus import Collection, connections

def count_assessments():
    """Get counts for each nima_assessment_overall value using query_iterator."""
    
    # Connect to Milvus
    connections.connect(uri="http://localhost:19530")
    db = MilvusDB()
    collection = Collection(db.collection_name)
    
    print("Counting nima_assessment_overall values...")
    print("="*50)
    
    counts = {}
    assessments = ["acceptable", "unacceptable"]
    
    # Count each assessment type using query_iterator
    for assessment in assessments:
        try:
            print(f"Counting {assessment} images...")
            
            filter_expr = f'nima_assessment_overall == "{assessment}"'
            
            iterator = collection.query_iterator(
                batch_size=1000,
                expr=filter_expr,
                output_fields=["id"]
            )
            
            count = 0
            batch_num = 0
            
            try:
                while True:
                    batch_results = iterator.next()
                    if not batch_results:
                        break
                    
                    batch_num += 1
                    count += len(batch_results)
                    
                    # Progress every 50 batches (50k records)
                    if batch_num % 50 == 0:
                        print(f"  Progress: {count:,} {assessment} images...")
            
            finally:
                iterator.close()
            
            counts[assessment] = count
            print(f"✓ {assessment}: {count:,}")
            
        except Exception as e:
            print(f"Error counting {assessment}: {e}")
            counts[assessment] = 0
    
    # Calculate totals
    total_counted = sum(counts.values())
    
    print(f"")
    print("FINAL COUNTS (NIMA Overall Assessment):")
    print("="*40)
    for assessment, count in counts.items():
        percentage = (count / total_counted * 100) if total_counted > 0 else 0
        print(f"{assessment.upper()}: {count:,} ({percentage:.1f}%)")
    
    print(f"")
    print(f"TOTAL IMAGES: {total_counted:,}")
    
    connections.disconnect("default")

if __name__ == "__main__":
    count_assessments()
