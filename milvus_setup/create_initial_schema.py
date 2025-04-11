from pymilvus import (
    connections,
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    Collection,
    utility
)

def create_kalliste_schema():
    # Connect to Milvus
    connections.connect(
        alias="default",
        host='localhost',
        port='19530'
    )

    collection_name = "kalliste_images"

    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection {collection_name}")

    # Define fields
    id_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )
    
    # dinov2_vector = FieldSchema(
    #     name="dinov2_vector",
    #     dtype=DataType.FLOAT_VECTOR,
    #     dim=1024  # Using DINOv2-L
    # )

    openclip_vector = FieldSchema(
        name="openclip_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=768  # Using ViT-L/14
    )

    # Fields from schema.sql - all made nullable
    image_file_path = FieldSchema(
        name="image_file_path",
        dtype=DataType.VARCHAR,
        max_length=500,
        is_nullable=False  # This should remain required
    )
    
    photoshoot = FieldSchema(
        name="photoshoot",
        dtype=DataType.VARCHAR,
        max_length=200,
        is_nullable=True
    )
    
    photoshoot_date = FieldSchema(
        name="photoshoot_date",
        dtype=DataType.VARCHAR,
        max_length=100,
        is_nullable=True
    )
    
    photoshoot_location = FieldSchema(
        name="photoshoot_location",
        dtype=DataType.VARCHAR,
        max_length=200,
        is_nullable=True
    )
    
    person_name = FieldSchema(
        name="person_name",
        dtype=DataType.VARCHAR,
        max_length=200,
        is_nullable=True
    )
    
    source_type = FieldSchema(
        name="source_type",
        dtype=DataType.VARCHAR,
        max_length=50,
        is_nullable=True
    )
    
    lr_rating = FieldSchema(
        name="lr_rating",
        dtype=DataType.INT64,
        is_nullable=True
    )
    
    lr_label = FieldSchema(
        name="lr_label",
        dtype=DataType.VARCHAR,
        max_length=100,
        is_nullable=True
    )
    
    image_date = FieldSchema(
        name="image_date",
        dtype=DataType.VARCHAR,
        max_length=100,
        is_nullable=True
    )
    
    region_type = FieldSchema(
        name="region_type",
        dtype=DataType.VARCHAR,
        max_length=50,
        is_nullable=True
    )
    
    nima_technical_score = FieldSchema(
        name="nima_technical_score",
        dtype=DataType.DOUBLE,
        is_nullable=True
    )
    
    nima_assessment_aesthetic = FieldSchema(
        name="nima_assessment_aesthetic",
        dtype=DataType.VARCHAR,
        max_length=500,
        is_nullable=True
    )
    
    nima_assessment_technical = FieldSchema(
        name="nima_assessment_technical",
        dtype=DataType.VARCHAR,
        max_length=500,
        is_nullable=True
    )
    
    nima_assessment_overall = FieldSchema(
        name="nima_assessment_overall",
        dtype=DataType.VARCHAR,
        max_length=500,
        is_nullable=True
    )
    
    nima_score_aesthetic = FieldSchema(
        name="nima_score_aesthetic",
        dtype=DataType.DOUBLE,
        is_nullable=True
    )
    
    nima_score_technical = FieldSchema(
        name="nima_score_technical",
        dtype=DataType.DOUBLE,
        is_nullable=True
    )
    
    nima_score_calc_average = FieldSchema(
        name="nima_score_calc_average",
        dtype=DataType.DOUBLE,
        is_nullable=True
    )
    
    kalliste_assessment = FieldSchema(
        name="kalliste_assessment",
        dtype=DataType.VARCHAR,
        max_length=500,
        is_nullable=True
    )
    
    record_creation_date = FieldSchema(
        name="record_creation_date",
        dtype=DataType.VARCHAR,
        max_length=100,
        is_nullable=True
    )
    
    all_tags = FieldSchema(
        name="all_tags",
        dtype=DataType.VARCHAR,
        max_length=4000,  # Long field to accommodate many tags
        is_nullable=True
    )

    # Create schema with all fields
    schema = CollectionSchema(
        fields=[
            id_field, 
            openclip_vector,
            image_file_path,
            photoshoot,
            photoshoot_date,
            photoshoot_location,
            person_name,
            source_type,
            lr_rating,
            lr_label,
            image_date,
            region_type,
            nima_assessment_aesthetic,
            nima_assessment_technical,
            nima_assessment_overall,
            nima_score_aesthetic,
            nima_score_technical,
            nima_score_calc_average,
            kalliste_assessment,
            record_creation_date,
            all_tags
        ],
        description="Kalliste image collection with CLIP embeddings"
    )

    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2
    )

    print(f"Created collection {collection.name}")
    
    # Clean up connection
    connections.disconnect("default")

if __name__ == "__main__":
    create_kalliste_schema()