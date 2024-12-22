# Model Weights Management in Kalliste

## Overview

Kalliste uses several AI models for different tasks. This document outlines how model weights are managed.

## Model Weight Locations

### Default Cache Locations
Most models use their default cache locations:

- **YOLO Models** (Object/Face Detection)
  - Location: `~/.cache/ultralytics`
  - Models: yolov8n.pt, yolov8n-face.pt
  - Managed by: ultralytics library
  - Note: Will download automatically on first use

- **BLIP2** (Image Captioning)
  - Location: `~/.cache/huggingface`
  - Model: Salesforce/blip2-opt-2.7b
  - Managed by: HuggingFace Transformers
  - Note: Will download automatically on first use

- **Orientation Model**
  - Location: `~/.cache/huggingface`
  - Model: LucyintheSky/pose-estimation-front-side-back
  - Managed by: HuggingFace Transformers
  - Note: Will download automatically on first use

### Custom Weight Location
Only one model requires special handling:

- **WD14 Tagger**
  - Location: `./weights/wd14/`
  - Files needed:
    - Model weights (downloaded via timm)
    - selected_tags.csv
  - Reason: Special handling required for tag processing
  - Note: Requires manual setup (see below)

## Setup Instructions

### Default Models
No manual setup required. The following will be downloaded automatically:
```bash
~/.cache/ultralytics/
  └── yolov8n.pt          # General object detection
  └── yolov8n-face.pt     # Face detection

~/.cache/huggingface/
  └── models/
      └── blip2-opt-2.7b/     # Image captioning
      └── pose-estimation/     # Pose orientation
```

### WD14 Tagger Setup
1. Create the weights directory:
   ```bash
   mkdir -p ./weights/wd14
   ```

2. Copy the required files:
   ```bash
   cp path/to/selected_tags.csv ./weights/wd14/
   ```

## Code Structure

### Weight Path References
- Models using default caches should NOT specify cache directories
- Only WD14 should specify its weight location explicitly

Example (taggers.py):
```python
# Only WD14 needs explicit path
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
WD14_CACHE = WEIGHTS_DIR / "wd14"

# Other models use default cache
orientation_model = AutoModelForImageClassification.from_pretrained(
    "LucyintheSky/pose-estimation-front-side-back"  # No cache_dir specified
)

# WD14 needs explicit path
tags_path = WD14_CACHE / "selected_tags.csv"
```

## Troubleshooting

Common issues:

1. YOLO weights downloading to wrong location:
   - Check no explicit paths are set in crop_processor.py
   - Let ultralytics manage its own cache

2. HuggingFace models downloading to ./weights:
   - Remove cache_dir parameters from .from_pretrained() calls
   - Exception: Keep cache_dir for WD14 setup

3. Missing WD14 files:
   - Verify ./weights/wd14 directory exists
   - Check selected_tags.csv is present

## Best Practices

1. Use default caches when possible:
   - Easier maintenance
   - Consistent with library defaults
   - Shared cache across projects

2. Only override for special cases:
   - Document the reason
   - Keep overrides minimal
   - Centralize custom path definitions

3. Code comments:
   - Mark explicit cache directories as exceptions
   - Explain why special handling is needed
   - Reference this documentation