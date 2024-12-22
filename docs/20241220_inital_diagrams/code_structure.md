# Kalliste Code Structure

```mermaid
classDiagram
    class Detectors {
        +base.py
        +yolo_detector.py
    }
    
    class Models {
        +exported_image.py
    }
    
    class Processors {
        +crop_processor.py
        +detection_pipeline.py
        +region_processor.py
        +taggers.py
    }

    class Utils {
        +utility functions
    }

    Processors --> Detectors : uses
    Processors --> Models : processes
    Detectors --> Models : detects
    Processors --> Utils : uses
    Detectors --> Utils : uses

    namespace kalliste {
        class DetectionPipeline {
            +process()
            +detect()
            +tag()
        }
        class YoloDetector {
            +detect()
            +load_model()
        }
        class RegionProcessor {
            +process_regions()
        }
        class CropProcessor {
            +process_crops()
        }
        class Taggers {
            +tag_detections()
        }
    }
```

Note: This diagram represents an older version of the codebase that used Pixeltable. The current implementation uses ChromaDB.