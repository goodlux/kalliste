# Kalliste Architecture Overview

## Core Processing Pipeline

The main Kalliste pipeline processes images in several stages:

```mermaid
graph TD
    A[Input Directory] --> B[Image Detection]
    B --> C[Region Processing]
    C --> D[SDXL Cropping]
    D --> E[Multi-Tagger Pipeline]
    E --> F[Metadata Writing]
    E --> G[Vector Database]
    F --> H[Output Directory]
    G --> I[ChromaDB Storage]
```

## Detection and Region Flow

```mermaid
graph TD
    A[Original Image] --> B[YOLO Detection]
    B --> C[Face Detection]
    B --> D[Person Detection]
    C --> E[Region Creation]
    D --> E
    E --> F[Region Expansion]
    F --> G[SDXL Ratio Check]
    G --> H[Valid Region]
```

## Tagger Pipeline

```mermaid
graph TD
    A[Cropped Image] --> B[WD14 Tagger]
    A --> C[BLIP2 Caption]
    A --> D[NIMA Quality]
    A --> E[Orientation]
    A --> F[CLIP Embedding]
    B --> G[Tag Collection]
    C --> G
    D --> G
    E --> G
    F --> H[ChromaDB]
    G --> I[XMP Metadata]
    G --> J[Caption File]
```

## Metadata Flow

```mermaid
graph TD
    A[Region Tags] --> B[Kalliste Tags]
    B --> C[XMP Metadata]
    B --> D[Caption File]
    B --> E[ChromaDB Entry]
    C --> F[Output Image]
    D --> G[.txt File]
    E --> H[Vector Database]
```

## Training Data Preparation

```mermaid
graph LR
    A[Processed Images] --> B[ChromaDB]
    B --> C[Vector Search]
    C --> D[Similar Group Selection]
    D --> E[Diversity Filter]
    E --> F[Training Dataset]
    F --> G[Final Training Directory]
```

## Component Relationships

```mermaid
graph TD
    subgraph Image Processing
        A[Original Image] --> B[Detection Models]
        B --> C[Region Processing]
    end
    
    subgraph Analysis
        C --> D[Tagger Pipeline]
        D --> E[Quality Assessment]
    end
    
    subgraph Storage
        E --> F[Metadata Storage]
        E --> G[Vector Database]
    end
    
    subgraph Training
        G --> H[Dataset Creation]
        F --> H
        H --> I[Training Data]
    end
```

## Key Components

- **Detection**: Uses YOLO models for initial detection of regions of interest
- **Region Processing**: Handles expansion, ratio correction, and validation
- **Tagger Pipeline**: Multiple models for analysis and metadata generation
  - WD14: Image classification and tagging
  - BLIP2: Caption generation
  - NIMA: Quality assessment
  - Orientation: Pose/view detection
  - CLIP: Vector embeddings
- **Storage**:
  - Filesystem: Organized output directories with metadata
  - ChromaDB: Vector database for similarity search
  - XMP: Embedded metadata in images
- **Training Preparation**:
  - Vector similarity search
  - Diversity-based selection
  - Dataset organization

## Model Management

- Centralized model registry
- Asynchronous model loading
- Device optimization (CUDA/MPS/CPU)
- Cached model downloads
