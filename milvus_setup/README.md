# Milvus Schema Setup Instructions

To update the Milvus schema with the new `all_tags` field, follow these steps:

1. Make sure the Milvus server is running (check with `docker ps`)
2. Activate your Python environment that has pymilvus installed
3. Run the schema creation script:

```bash
cd /Users/rob/repos/kalliste
python milvus_setup/create_initial_schema.py
```

This will:
- Drop the existing collection
- Create a new collection with the updated schema including the `all_tags` field
- Set up the proper vector index configurations

Note that dropping the collection will delete all existing data. If you want to preserve data, you would need to:
1. Export the existing data first
2. Create the new schema
3. Import the data back with the additional field

The new `all_tags` field will be populated from:
- KallisteLrTags
- KallisteWd14Tags 
- KallisteTags