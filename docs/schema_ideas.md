# Schema Ideas from Previous Implementation

These are concepts from the previous Pixeltable implementation that might be useful to consider when designing the ChromaDB collections:

## Core Identifiers and Paths
- project_id
- shoot_event
- media_type ('source' or 'crop')
- capture_date
- original_filename
- original_path
- original_timestamp

## Training Text Fields
- auto_caption
- lr_keywords (list)
- auto_tags (list)

## AI Detection Fields
- pose_tags (list)
- clothing_tags (list)
- lookalike_tags (list)
- detection_tags (list)

## Image Properties
- type ('source', 'person_crop', 'face_crop')
- orientation ('portrait', 'landscape', 'square')

## Technical Metadata
- processing_metadata (processing history, crops, etc)
- created_at

Note: This schema is for reference only. The ChromaDB implementation should be designed based on current requirements and ChromaDB's strengths/limitations.