# Kalliste Project Guide

## Build & Testing
- Run individual tests: `python tests/<test_number>_test_dng.py` or `python scripts/test_<name>.py`
- Test detection module: `python kalliste/test_detect.py --image_path <path>`
- Process pipeline: `python scripts/test_pipeline.py` 
- No formal test runner - tests are standalone Python scripts

## Code Style Guidelines

### Imports & Structure
- Standard library → third-party → project imports (separated by blank lines)
- Relative imports with explicit dots (e.g., `from ..region import Region`)
- Import specific classes/functions directly
- Classes separated by two blank lines, methods by one

### Naming & Typing
- Classes: PascalCase (`BaseDetector`, `ModelRegistry`)
- Functions/methods: snake_case (`detect`, `tag_pillow_image`)
- Private methods: prefixed with underscore (`_validate_image_path`)
- Constants: UPPER_SNAKE_CASE (`KALLISTE_DATA_DIR`)
- Use type hints for all parameters and return values

### Error Handling & Documentation
- Catch specific exception types when possible
- Log errors with context details
- Google-style docstrings with Args and Returns sections
- 4-space indentation, ~100 char line length
- Follow existing patterns when extending functionality

## Next Steps

### Milvus Queries
Implement query functionality for the Milvus database:

1. **Photoshoot filtering**:
   - Create query to filter by `photoshoot_name`
   - Add filtering for accepted images only (exclude rejects)

2. **Image diversity sampling**:
   - Implement algorithm to select ~5000 distinct, high-quality images
   - Use vector embeddings to ensure diversity (avoid nearly identical frames from video)
   - Prioritize images with higher NIMA scores
   - Cluster similar images and sample representatives

3. **Query interface**:
   - Create functions to search by text using OpenCLIP embeddings
   - Add filtering by tags from the `all_tags` field
   - Support hybrid searches combining vector similarity and metadata filters