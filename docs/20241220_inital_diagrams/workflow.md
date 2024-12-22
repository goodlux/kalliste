# Kalliste Workflow

## Core Processing Pipeline

```mermaid
flowchart TD
    subgraph SourceProcessing["1. Source Processing"]
        A[DNG Files] --> B[Quality Service]
        B -->|Basic Checks| C{Quality Types}
        C -->|Technical| D[Blur Detection]
        C -->|Exposure| E[Exposure Check]
        D & E -->|Write| F[DNG Metadata]
    end

    subgraph Selection["2. Lightroom Selection"]
        F -->|Import Changes| G[Lightroom]
        G -->|Review| H{Quality Check}
        H -->|Bad| I[Delete/Archive]
        H -->|Good| J[Face Detection]
        J -->|Manual| K[Additional Tags]
        K -->|Export| L[Named Export Folder]
    end

    subgraph Cropping["3. Crop Processing"]
        L --> M{Crop Type}
        M -->|Person| N[Person Crop]
        M -->|Face| O[Face Crop]
        N --> P{SDXL Size Check}
        O --> Q{Face Size Check}
        P -->|Too Small| R[Discard]
        Q -->|Too Small| R
        P -->|Good| S[Save Person Crop]
        Q -->|Good| T[Save Face Crop]
    end

    subgraph Analysis["4. ML Analysis"]
        S & T --> U[Analysis Service]
        U -->|YOLO| V[Pose Detection]
        U -->|Vision Model| W[Image Captioning]
        U -->|Custom| X[Other Detections]
        V & W & X --> Y[Crop Metadata]
    end

    subgraph DataStorage["5. Data Management"]
        Y --> Z[ChromaDB]
        Z --> AA[Create Embeddings]
        Y --> AB[Sidecar Files]
        AA & AB --> AC[Training Server]
    end

    subgraph Training["6. Training Process"]
        AC --> AD[Create LoRA]
        AD --> AE[Training Manifest]
        AE --> AF{Evaluate Results}
        AF -->|Needs Improvement| A
        AF -->|Success| AG[Final Model]
    end

    %% Add some styling
    classDef process fill:#e1f5fe,stroke:#01579b;
    classDef decision fill:#fff3e0,stroke:#ff6f00;
    classDef storage fill:#e8f5e9,stroke:#2e7d32;
    
    class B,D,E,G,J,K,N,O,U,V,W,X process;
    class H,P,Q,AF decision;
    class A,F,L,Y,Z,AC,AE storage;
```

## Workflow Overview

1. **Source Processing**: Basic quality checks on DNG files
   - Blur detection
   - Exposure checks
   - Write results to DNG metadata

2. **Lightroom Selection**: Manual curation and enhancement
   - Review quality check results
   - Face detection
   - Manual tagging
   - Export selected images

3. **Crop Processing**: Automated cropping for training
   - Person crops with SDXL size validation
   - Face crops with size validation
   - Discard undersized crops

4. **ML Analysis**: Process cropped images
   - Pose detection
   - Image captioning
   - Custom detections
   - Store metadata with crops

5. **Data Management**: Prepare for training
   - Add to ChromaDB with embeddings
   - Generate sidecar files
   - Transfer to training server

6. **Training Process**: Create and evaluate LoRA
   - Train model
   - Generate manifest
   - Evaluate results
   - Iterate if needed
