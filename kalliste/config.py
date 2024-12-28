from pathlib import Path
from pathlib import Path

# Cache directories
CACHE_DIR = Path.home() / ".cache"
YOLO_CACHE_DIR = CACHE_DIR / "ultralytics"
HF_CACHE_DIR = CACHE_DIR / "huggingface" / "hub"


# Core model configurations - all in default cache locations
MODELS = {
    # Currently we are only using YOLO models for detection
    "detection": {
        "yolo": {
            "model_id": "yolo",
            "file": "yolov11x.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
        },
        "yolo-face": {
            "model_id": "yolo-face",
            "file": "yolov11m-face.pt",
            "url": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"
        }
    },
    # Currently any other models MUST be available on Hugging Face Hub.
    "classification": {
        "wd14": {
            "model_id": "wd14",
            "hf_path": "SmilingWolf/wd-vit-large-tagger-v3",
            "files": ["selected_tags.csv"]
        },
        "blip2": {
            "model_id": "blip2",
            "hf_path": "Salesforce/blip2-opt-2.7b",
            "files": []
        },
        "orientation": {
            "model_id": "orientation",
            "hf_path": "LucyintheSky/pose-estimation-front-side-back",
            "files": []
        }
    }
}

# TODO: Remove these from the codebase, instead use the MODELS config above. 
# Use the model IDs from MODELS dictionary
BLIP2_MODEL_ID = MODELS["classification"]["blip2"]
ORIENTATION_MODEL_ID = MODELS["classification"]["orientation"]
WD14_MODEL_ID = MODELS["classification"]["wd14"]



# # Detection configurations
# DETECTION_CONFIG = {
#     # Default confidence threshold for detectors
#     "confidence_threshold": 0.5,
    
#     # Default detection types to look for (if not specified)
#     "default_detection_types": ["person", "face"]
# }
