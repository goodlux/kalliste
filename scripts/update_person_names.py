#!/usr/bin/env python3
"""
Update person_name field in Milvus for records where it's empty but can be inferred from photoshoot name.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import Collection, connections
from kalliste.db.milvus_db import MilvusDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_person_names(person_name: str, photoshoot_contains: str):
    """
    Update person_name where it's empty and photoshoot contains a specific string.

    Args:
        person_name: The person name to set
        photoshoot_contains: String that must be in photoshoot field
    """
    # Connect to Milvus
    connections.connect(uri="http://localhost:19530")
    db = MilvusDB()
    collection = Collection(db.collection_name)

    # Build filter expression
    # Note: Milvus doesn't have "contains" - we need to query all empty person_names
    # and filter in Python
    filter_expr = 'person_name == ""'

    # Get all fields we need (including vectors for re-insertion)
    output_fields = ["*"]  # Get all fields

    logger.info(f"Querying for records with empty person_name...")

    # Use query_iterator for large datasets
    batch_size = 1000
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr=filter_expr,
        output_fields=output_fields
    )

    records_to_update = []
    total_checked = 0

    try:
        while True:
            batch_results = iterator.next()
            if not batch_results:
                break

            total_checked += len(batch_results)

            # Filter for records where photoshoot contains our string
            for record in batch_results:
                photoshoot = record.get('photoshoot', '')
                if photoshoot_contains.lower() in photoshoot.lower():
                    records_to_update.append(record)

            logger.info(f"Checked {total_checked} records, found {len(records_to_update)} to update...")

    finally:
        iterator.close()

    if not records_to_update:
        logger.info(f"No records found where person_name is empty and photoshoot contains '{photoshoot_contains}'")
        return 0

    logger.info(f"Found {len(records_to_update)} records to update")

    # Now update them in batches
    # We need to delete and re-insert
    updated_count = 0
    batch_size = 100

    for i in range(0, len(records_to_update), batch_size):
        batch = records_to_update[i:i+batch_size]

        # Extract IDs to delete
        ids_to_delete = [str(record['id']) for record in batch]

        # Delete old records
        id_list = ",".join(ids_to_delete)
        delete_expr = f"id in [{id_list}]"
        collection.delete(delete_expr)

        # Update person_name in records and remove id field
        updated_batch = []
        for record in batch:
            record['person_name'] = person_name
            # Remove the id field - it's auto-generated
            if 'id' in record:
                del record['id']
            updated_batch.append(record)

        # Re-insert updated records
        collection.insert(updated_batch)

        updated_count += len(batch)
        logger.info(f"Updated {updated_count}/{len(records_to_update)} records...")

    # Flush to ensure changes are persisted
    collection.flush()

    connections.disconnect("default")
    logger.info(f"Successfully updated {updated_count} records")
    return updated_count

def main():
    """Update person_name to 'MiSoLux' where person_name is empty and photoshoot contains 'MiSoLux'"""

    logger.info("=" * 60)
    logger.info("Updating person names in Milvus")
    logger.info("=" * 60)

    # Update MiSoLux
    count = update_person_names("MiSoLux", "MiSoLux")

    logger.info(f"\nTotal records updated: {count}")

if __name__ == "__main__":
    main()