from pathlib import Path
from pathlib import Path

# Cache directories
CACHE_DIR = Path.home() / ".cache"
YOLO_CACHE_DIR = CACHE_DIR / "ultralytics"
HF_CACHE_DIR = CACHE_DIR / "huggingface" / "hub"

# Core model configurations - all in default cache locations
MODELS = {
    "detection": {
        # In ~/.cache/ultralytics
        "yolo": "yolov11x.pt"
    },
    "classification": {
        # All in ~/.cache/huggingface/hub
        "blip2": "Salesforce/blip2-opt-2.7b",
        "orientation": "LucyintheSky/pose-estimation-front-side-back",
        "wd14": "SmilingWolf/wd-vit-large-tagger-v3"
    }
}

# Detection configurations
DETECTION_CONFIG = {
    # Default confidence threshold for detectors
    "confidence_threshold": 0.5,
    
    # Default detection types to look for (if not specified)
    "default_detection_types": ["person", "face"]
}


# Huggingface Models - Using default HF cache
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
ORIENTATION_MODEL_ID = "LucyintheSky/pose-estimation-front-side-back"

# WD14 Configuration
WD14_MODEL_ID = "hf_hub:SmilingWolf/wd-vit-large-tagger-v3"
PROJECT_ROOT = Path(__file__).parent.parent
WD14_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "wd14"
WD14_TAGS_FILE = WD14_WEIGHTS_DIR / "selected_tags.csv"

# Only WD14 needs explicit directory management
WD14_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)