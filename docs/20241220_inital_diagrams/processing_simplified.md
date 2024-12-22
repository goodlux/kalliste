# Simplified Processing Flow

This diagram shows the core image processing flow, focusing on metadata preservation throughout the process.

```mermaid
classDiagram
    class ExportedImage {
        +Path source_path
        +Dict lr_metadata
        +String shoot_name
        +DateTime export_date
        +List[Region] face_regions
        +extract_face_regions()
        +get_quality_metrics()
    }

    class Region {
        +Tuple coordinates
        +String region_type
        +validate_sdxl_size()
        +get_crop_dimensions()
    }

    class CropProcessor {
        +ultralytics_model
        +create_person_crops()
        +validate_crop_size()
        +save_crop_with_metadata()
    }

    class ImageCrop {
        +ExportedImage source
        +CropType type
        +Region region
        +Path crop_path
        +Bool valid_for_sdxl
        +Dict quality_metrics
        +Dict metadata
        +get_orientation()
        +save_with_metadata()
    }

    class MLAnalyzer {
        +ultralytics_model
        +detect_orientation()
        +generate_caption()
        +add_to_metadata()
    }

    ExportedImage "1" *-- "many" Region
    ExportedImage --> CropProcessor : sends to
    Region --> ImageCrop : creates
    CropProcessor --> ImageCrop : processes
    ImageCrop --> MLAnalyzer : analyzed by
    MLAnalyzer --> ImageCrop : updates metadata
```

## Processing Steps

1. **ExportedImage**
   - Handles reading of exported images from Lightroom
   - Extracts face regions from existing metadata
   - Gets basic quality metrics if needed

2. **Region**
   - Represents either a face region from Lightroom or a person crop
   - Validates if region meets SDXL size requirements
   - Provides dimensions for cropping

3. **CropProcessor**
   - Uses ultralytics for person detection
   - Creates and validates crops
   - Preserves metadata during crop creation

4. **ImageCrop**
   - Represents a processed crop with its metadata
   - Maintains link to source image
   - Handles saving crop with preserved metadata

5. **MLAnalyzer**
   - Detects orientation using ultralytics
   - Generates captions if needed
   - Adds analysis results to image metadata
