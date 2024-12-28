```mermaid
classDiagram
    %% Core Image Class
    class KallisteImage {
        +str original_path
        +KallisteImage parent
        +List~KallisteImage~ children
        +Dict metadata
        +Dict kalliste_tags
        +str detection_type
        +str processing_stage
        +from_parent(parent, detection_type)
        +update_kalliste_tags(tags)
        +get_processing_pipeline()
        +export(output_dir)
        +is_processed()
    }

    %% Main Processor
    class ImageProcessor {
        +Detector detector
        +process_image(path)
    }

    %% Image Processing Components
    namespace image_utils {
        class Detector {
            +detect(image_path)
            +get_detections()
        }
        class Resizer {
            +resize(image, target_size)
        }
        class RegionProcessor {
            +process_region(image, detection)
        }
        class CropProcessor {
            +process_crop(image, region)
        }
    }

    %% Metadata Handling
    namespace metadata {
        class MetadataCopier {
            +extract_original(path)
            +write_metadata(path, metadata)
        }
        class MetadataEnricher {
            +combine_metadata(original, kalliste_tags)
            +create_xmp_namespace()
        }
        class MetadataSchemas {
            +KALLISTE_XMP_NAMESPACE
            +KALLISTE_TAG_DEFINITIONS
        }
    }

    %% Processing Rules and Types
    namespace processing {
        class ExpansionRules {
            +get_expansion_factor(detection_type)
            +apply_rules(region, detection_type)
        }
        class ProcessingPipeline {
            +List~Processor~ processors
            +get_pipeline(detection_type)
            +run_pipeline(image)
        }
    }

    %% Relationships
    ImageProcessor --> KallisteImage : creates
    ImageProcessor --> Detector : uses
    KallisteImage --> KallisteImage : parent/child
    KallisteImage --> ProcessingPipeline : gets
    ProcessingPipeline --> RegionProcessor : contains
    ProcessingPipeline --> CropProcessor : contains
    ProcessingPipeline --> Resizer : contains
    RegionProcessor --> ExpansionRules : uses
    KallisteImage --> MetadataCopier : uses
    KallisteImage --> MetadataEnricher : uses
    MetadataEnricher --> MetadataSchemas : references

    ```
    