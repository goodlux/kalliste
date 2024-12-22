# Post-Export Processing Flow

This diagram shows the flow of image processing after export from Lightroom.

```mermaid
classDiagram
    class ExportedImage {
        +Path source_path
        +Dict lr_metadata
        +String shoot_name
        +DateTime export_date
        +analyze_quality()
    }

    class CropProcessor {
        +detect_faces()
        +detect_persons()
        +create_crops()
        +validate_crop_size()
    }

    class ImageCrop {
        +ExportedImage source
        +CropType type
        +Tuple coordinates
        +Path crop_path
        +Bool valid_for_sdxl
        +Dict quality_metrics
    }

    class MLProcessor {
        +analyze_pose()
        +generate_caption()
        +run_detections()
        +create_embedding()
    }

    class AnalysisResult {
        +ImageCrop crop
        +Dict pose_data
        +String caption
        +Dict detections
        +Array embedding
        +save_metadata()
        +generate_sidecar()
    }

    class ChromaDBManager {
        +add_image()
        +find_similar()
        +get_diverse_set()
    }

    ExportedImage --> CropProcessor : sends to
    CropProcessor --> ImageCrop : creates
    ImageCrop --> MLProcessor : processed by
    MLProcessor --> AnalysisResult : produces
    AnalysisResult --> ChromaDBManager : stored in
```

## Processing Steps

1. **ExportedImage**: Represents a PNG exported from Lightroom
   - Contains original metadata from Lightroom
   - Identifies shoot based on folder naming
   - Can run basic quality analysis if needed

2. **CropProcessor**: Handles detection and cropping
   - Detects faces and persons in the image
   - Creates appropriate crops
   - Validates crops against SDXL size requirements

3. **ImageCrop**: Represents a cropped version
   - Links back to source image
   - Contains crop coordinates and type
   - Tracks SDXL validity
   - Includes quality metrics for the crop

4. **MLProcessor**: Runs ML analysis on crops
   - Pose detection
   - Captioning
   - Custom detections
   - Embedding generation

5. **AnalysisResult**: Stores analysis results
   - Contains all ML processing results
   - Handles metadata storage
   - Generates training sidecar files

6. **ChromaDBManager**: Manages similarity database
   - Stores images and embeddings
   - Provides similarity search
   - Helps create diverse training sets
