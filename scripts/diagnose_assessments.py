#!/usr/bin/env python3
"""
Comprehensive diagnostic script to check what assessment values are actually in the database
and test the cleanup logic to find the bug.
"""
import sys
from pathlib import Path
from collections import Counter

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from kalliste.db.milvus_db import MilvusDB
from pymilvus import Collection, connections

def check_assessment_values():
    """Check what assessment values actually exist in the database."""
    
    connections.connect(uri="http://localhost:19530")
    db = MilvusDB()
    collection = Collection(db.collection_name)
    
    print("COMPREHENSIVE ASSESSMENT DIAGNOSTIC")
    print("="*60)
    
    # Get sample records
    sample_results = collection.query(
        expr='person_name == "NaGiLux"',
        output_fields=["id", "nima_assessment_technical", "nima_assessment_overall", "nima_assessment_aesthetic"],
        limit=50
    )
    
    print(f"Sample records (first 50 NaGiLux images):")
    print("-" * 80)
    
    logic_errors = []
    
    for i, result in enumerate(sample_results):
        print(f"Record {i+1}:")
        tech = result.get('nima_assessment_technical', 'NULL')
        overall = result.get('nima_assessment_overall', 'NULL')
        aesthetic = result.get('nima_assessment_aesthetic', 'NULL')
        
        print(f"  Technical: '{tech}'")
        print(f"  Overall:   '{overall}'")
        print(f"  Aesthetic: '{aesthetic}'")
        
        # Test the logic on this record
        would_delete = (tech != "high_quality") and (overall != "acceptable")
        should_keep = (tech == "high_quality") or (overall == "acceptable")
        
        print(f"  Logic test: would_delete={would_delete}, should_keep={should_keep}")
        
        if would_delete and should_keep:
            print(f"  ⚠️  LOGIC ERROR: This record would be deleted but should be kept!")
            logic_errors.append((tech, overall))
        elif would_delete:
            print(f"  ✅ Would correctly DELETE (poor quality)")
        else:
            print(f"  ✅ Would correctly KEEP (good quality)")
        print()
    
    if logic_errors:
        print(f"\n❌ FOUND {len(logic_errors)} LOGIC ERRORS in sample!")
        print("These combinations were marked for deletion but should be kept:")
        for tech, overall in set(logic_errors):
            print(f"  technical='{tech}', overall='{overall}'")
    else:
        print("\n✅ No logic errors found in sample")
    
    # Get larger sample to find unique values
    print(f"\nGetting larger sample for value analysis...")
    large_sample = collection.query(
        expr='person_name == "NaGiLux"',
        output_fields=["nima_assessment_technical", "nima_assessment_overall", "nima_assessment_aesthetic"],
        limit=5000
    )
    
    # Collect unique values
    tech_values = set()
    overall_values = set()
    aesthetic_values = set()
    
    for result in large_sample:
        tech = result.get('nima_assessment_technical')
        overall = result.get('nima_assessment_overall')
        aesthetic = result.get('nima_assessment_aesthetic')
        
        if tech:
            tech_values.add(tech)
        if overall:
            overall_values.add(overall)
        if aesthetic:
            aesthetic_values.add(aesthetic)
    
    print(f"\nUNIQUE VALUES ANALYSIS:")
    print("-" * 40)
    print(f"Technical assessment values ({len(tech_values)} unique): {sorted(tech_values)}")
    print(f"Overall assessment values ({len(overall_values)} unique): {sorted(overall_values)}")
    print(f"Aesthetic assessment values ({len(aesthetic_values)} unique): {sorted(aesthetic_values)}")
    
    # Count occurrences
    print(f"\nVALUE COUNTS (from {len(large_sample)} records):")
    print("-" * 40)
    
    tech_counts = Counter(result.get('nima_assessment_technical') for result in large_sample)
    overall_counts = Counter(result.get('nima_assessment_overall') for result in large_sample)
    
    print("Technical assessment counts:")
    for value, count in tech_counts.most_common():
        print(f"  '{value}': {count:,}")
    
    print("\nOverall assessment counts:")
    for value, count in overall_counts.most_common():
        print(f"  '{value}': {count:,}")
    
    # Test the current filter logic against actual data
    print(f"\nFILTER LOGIC TEST ON ACTUAL DATA:")
    print("-" * 40)
    
    # Test current filter: (tech != "high_quality") and (overall != "acceptable")
    would_delete_count = 0
    should_keep_but_delete_count = 0
    correctly_delete_count = 0
    correctly_keep_count = 0
    
    print("Testing logic on first 100 records...")
    
    for i, result in enumerate(large_sample[:100]):
        tech = result.get('nima_assessment_technical')
        overall = result.get('nima_assessment_overall')
        
        # Current filter logic (what we're using to delete)
        would_delete = (tech != "high_quality") and (overall != "acceptable")
        
        # What we actually want (images that should be kept)
        should_keep = (tech == "high_quality") or (overall == "acceptable")
        
        if would_delete:
            would_delete_count += 1
            if should_keep:
                should_keep_but_delete_count += 1
                print(f"  ❌ Record {i+1}: tech='{tech}', overall='{overall}' - LOGIC ERROR!")
            else:
                correctly_delete_count += 1
        else:
            correctly_keep_count += 1
    
    print(f"\nRESULTS from 100 test records:")
    print(f"  Would delete: {would_delete_count}")
    print(f"  Would keep: {100 - would_delete_count}")
    print(f"  Logic errors (delete but should keep): {should_keep_but_delete_count}")
    print(f"  Correctly delete: {correctly_delete_count}")
    print(f"  Correctly keep: {correctly_keep_count}")
    
    if should_keep_but_delete_count > 0:
        print(f"\n❌ CRITICAL: {should_keep_but_delete_count} records would be incorrectly deleted!")
        print("The cleanup script has a logic error and should NOT be run.")
    else:
        print(f"\n✅ Logic looks correct on test sample.")
    
    # Show what the correct filter should be
    print(f"\nCORRECT FILTER ANALYSIS:")
    print("-" * 30)
    print("To DELETE (reject), we want:")
    print("  (technical != 'high_quality') AND (overall != 'acceptable')")
    print("")
    print("To KEEP (accept), we want:")
    print("  (technical == 'high_quality') OR (overall == 'acceptable')")
    print("")
    print("Current database values show:")
    print(f"  Technical values: {sorted(tech_values)}")
    print(f"  Overall values: {sorted(overall_values)}")
    
    connections.disconnect("default")

if __name__ == "__main__":
    check_assessment_values()
