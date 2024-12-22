from pathlib import Path
import os

# YOLO Configuration - Using default ultralytics cache
YOLO_PERSON_MODEL = 'yolov8n.pt'
YOLO_FACE_MODEL = 'yolov8n-face.pt'
YOLO_CACHE_DIR = Path(os.path.expanduser('~/.cache/ultralytics'))

# Huggingface Models - Using default HF cache
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
ORIENTATION_MODEL_ID = "LucyintheSky/pose-estimation-front-side-back"

# WD14 Configuration
WD14_MODEL_ID = "hf_hub:SmilingWolf/wd-vit-large-tagger-v3"
PROJECT_ROOT = Path(__file__).parent.parent
WD14_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "wd14"
WD14_TAGS_FILE = WD14_WEIGHTS_DIR / "selected_tags.csv"

# Ensure cache directories exist
YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
WD14_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)