"""Kalliste configuration."""

# Base data directory
KALLISTE_DATA_DIR = "/Volumes/m01/kalliste_data"

# Kalliste database settings
KALLISTE_DB_PATH = "/Volumes/m01/kalliste_data/kalliste_db/kalliste.sqlite3"

# Chroma database settings
CHROMA_DB_DIR = "/Volumes/m01/kalliste_data/chroma_db"


# Cache directories
CACHE_DIR = "~/.cache"
YOLO_CACHE_DIR = CACHE_DIR + "/ultralytics"
HF_CACHE_DIR =  CACHE_DIR + "/huggingface/hub"
NIMA_CACHE_DIR = CACHE_DIR + "/kalliste/nima"

# Core model configurations
MODELS = {
    # Currently we are only using YOLO models for detection
    "detection": {
        "yolo": {
            "model_id": "yolo",
            "file": "yolo11x.pt",
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
        },
        "nima": {
            "model_id": "nima",
            "urls": {
                "technical": "https://github.com/idealo/image-quality-assessment/raw/refs/heads/master/models/MobileNet/weights_mobilenet_technical_0.11.hdf5",
                "aesthetic": "https://github.com/idealo/image-quality-assessment/raw/refs/heads/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"
            },
            "files": {
                "technical": "technical_0.11.hdf5",
                "aesthetic": "aesthetic_0.07.hdf5"
            }
        },
    }
}

# Convenience references to model IDs
BLIP2_MODEL_ID = MODELS["classification"]["blip2"]
ORIENTATION_MODEL_ID = MODELS["classification"]["orientation"]
WD14_MODEL_ID = MODELS["classification"]["wd14"]