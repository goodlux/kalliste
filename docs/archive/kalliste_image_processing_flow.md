```mermaid
flowchart TD
    Start([Start]) --> LoadImage[Load Original Image]
    LoadImage --> CreateObj[Create KallisteImage Object]
    CreateObj --> ExtractMeta[Extract Original Metadata]
    ExtractMeta --> RunDetect[Run Detection]
    
    RunDetect --> HasDetections{Detections Found?}
    HasDetections -- No --> NextImage[Process Next Image]
    NextImage --> Start
    
    HasDetections -- Yes --> CreateCrops[Create Child Images for Each Detection]
    
    CreateCrops --> ProcessLoop[Process Each Child Image]
    
    subgraph ProcessingPipeline[Processing Pipeline]
        ProcessLoop --> ApplyRegion[Apply Region Processing]
        ApplyRegion --> ApplyCrop[Apply Cropping]
        ApplyCrop --> RunTaggers[Run Appropriate Taggers]
        RunTaggers --> UpdateMeta[Update Kalliste Tags]
    end
    
    UpdateMeta --> Resize[Resize Image]
    Resize --> CombineMeta[Combine Original & Kalliste Metadata]
    CombineMeta --> SaveImage[Save Processed Image]
    
    SaveImage --> More{More Detections?}
    More -- Yes --> ProcessLoop
    More -- No --> NextImage
    
    style ProcessingPipeline fill:#f5f5f5,stroke:#333,stroke-width:2px

    ```