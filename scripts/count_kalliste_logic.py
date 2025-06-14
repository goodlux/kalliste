#!/usr/bin/env python3
"""
Count images using the actual Kalliste accept/reject logic:
- ACCEPT if: nima_assessment_technical == "high_quality" OR nima_assessment_overall == "acceptable"
- REJECT everything else
"""
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB
from pymilvus import Collection, connections

def count_kalliste_logic():
    """Count using actual Kalliste accept/reject logic."""
    
    # Connect to Milvus
    connections.connect(uri="http://localhost:19530")
    db = MilvusDB()
    collection = Collection(db.collection_name)
    
    print("Counting using ACTUAL Kalliste logic...")
    print("Accept if: technical=='high_quality' OR overall=='acceptable'")
    print("="*60)
    
    # Count images that should be ACCEPTED
    print("Counting ACCEPTED images...")
    accept_expr = '(nima_assessment_technical == "high_quality") or (nima_assessment_overall == "acceptable")'
    
    iterator = collection.query_iterator(
        batch_size=1000,
        expr=accept_expr,
        output_fields=["id"]
    )
    
    accept_count = 0
    batch_num = 0
    
    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
            
            batch_num += 1
            accept_count += len(batch_results)
            
            if batch_num % 50 == 0:
                print(f"  Progress: {accept_count:,} accepted images...")
    
    finally:
        iterator.close()
    
    print(f"✓ ACCEPTED: {accept_count:,}")
    
    # Count images that should be REJECTED  
    print("Counting REJECTED images...")
    reject_expr = '(nima_assessment_technical != "high_quality") and (nima_assessment_overall != "acceptable")'
    
    iterator = collection.query_iterator(
        batch_size=1000,
        expr=reject_expr,
        output_fields=["id"]
    )
    
    reject_count = 0
    batch_num = 0
    
    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
            
            batch_num += 1
            reject_count += len(batch_results)
            
            if batch_num % 50 == 0:
                print(f"  Progress: {reject_count:,} rejected images...")
    
    finally:
        iterator.close()
    
    print(f"✓ REJECTED: {reject_count:,}")
    
    # Calculate totals
    total_counted = accept_count + reject_count
    
    print(f"")
    print("FINAL COUNTS (Kalliste Logic):")
    print("="*35)
    accept_pct = (accept_count / total_counted * 100) if total_counted > 0 else 0
    reject_pct = (reject_count / total_counted * 100) if total_counted > 0 else 0
    
    print(f"ACCEPTED: {accept_count:,} ({accept_pct:.1f}%)")
    print(f"REJECTED: {reject_count:,} ({reject_pct:.1f}%)")
    print(f"")
    print(f"TOTAL IMAGES: {total_counted:,}")
    
    # Show breakdown of accept reasons
    print(f"")
    print("ACCEPT BREAKDOWN:")
    print("="*20)
    
    # High quality technical (regardless of overall)
    high_tech_expr = 'nima_assessment_technical == "high_quality"'
    iterator = collection.query_iterator(
        batch_size=1000,
        expr=high_tech_expr,
        output_fields=["id"]
    )
    
    high_tech_count = 0
    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
            high_tech_count += len(batch_results)
    finally:
        iterator.close()
    
    # Acceptable overall (but not high technical)
    acceptable_not_high_expr = '(nima_assessment_overall == "acceptable") and (nima_assessment_technical != "high_quality")'
    iterator = collection.query_iterator(
        batch_size=1000,
        expr=acceptable_not_high_expr,
        output_fields=["id"]
    )
    
    acceptable_not_high_count = 0
    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break
            acceptable_not_high_count += len(batch_results)
    finally:
        iterator.close()
    
    print(f"High technical quality: {high_tech_count:,}")
    print(f"Acceptable overall (not high tech): {acceptable_not_high_count:,}")
    print(f"Total accepted: {high_tech_count + acceptable_not_high_count:,}")
    
    connections.disconnect("default")

if __name__ == "__main__":
    count_kalliste_logic()
