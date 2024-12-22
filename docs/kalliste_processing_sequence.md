sequenceDiagram
    participant Client
    participant ImageProcessor
    participant KallisteImage
    participant Detector
    participant Pipeline
    participant MetadataHandler

    Client->>ImageProcessor: process_image(path)
    ImageProcessor->>KallisteImage: create root image
    KallisteImage->>MetadataHandler: extract_original_metadata()
    
    ImageProcessor->>Detector: detect(path)
    Detector-->>ImageProcessor: detections[]

    loop For each detection
        ImageProcessor->>KallisteImage: from_parent(root, detection_type)
        KallisteImage->>Pipeline: get_processing_pipeline()
        
        loop For each processor
            Pipeline->>KallisteImage: process(image)
            KallisteImage->>KallisteImage: update_kalliste_tags()
        end

        KallisteImage->>MetadataHandler: combine_metadata()
        KallisteImage->>KallisteImage: export()
    end

    ImageProcessor-->>Client: processed_images[]