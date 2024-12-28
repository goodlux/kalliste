"""
Model Registry for Kalliste

Handles initialization and access to AI models.
Models are initialized once and stored in the registry for reuse.
Runtime parameters (like confidence thresholds) can be adjusted per-call.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
import timm
import torch
import torchvision.transforms as T
from ultralytics import YOLO

from .config import MODELS, YOLO_CACHE_DIR, HF_CACHE_DIR
from .model_download_manager import ModelDownloadManager

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton registry for initialized models.
    
    Models are initialized once when first needed and stored for reuse.
    Each model can be configured at runtime without reinitialization.
    
    The registry maintains two types of models:
    - Detection models (YOLO)
    - Classification models (from HuggingFace)
    """
    
    _instance = None
    _initialized = False
    _models: Dict[str, Any] = {}
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the model registry and download all required models."""
        async with cls._lock:
            if cls._initialized:
                return
                
            try:
                # First ensure all models are downloaded
                downloader = ModelDownloadManager()
                await downloader.download_all()
                
                # Initialize detection models
                await cls._initialize_detection_models()
                
                # Initialize classification models
                await cls._initialize_classification_models()
                
                cls._initialized = True
                logger.info("Model registry initialization complete")
                
            except Exception as e:
                logger.error(f"Failed to initialize model registry: {e}")
                await cls.cleanup()
                raise

    @classmethod
    async def _initialize_detection_models(cls) -> None:
        """Initialize all detection (YOLO) models."""
        for model_name, model_config in MODELS["detection"].items():
            try:
                model_path = YOLO_CACHE_DIR / model_config["file"]
                model = YOLO(str(model_path))
                cls._models[model_config["model_id"]] = {
                    "model": model,
                    "type": "detection"
                }
                logger.info(f"Initialized detection model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
                raise

    @classmethod
    async def _initialize_classification_models(cls) -> None:
        """Initialize all classification models."""
        for model_name, model_config in MODELS["classification"].items():
            try:
                # Special handling for WD14 tagger which needs model and transform
                if model_name == "wd14":
                    model = timm.create_model(
                        f"hf_hub:{model_config['hf_path']}", 
                        pretrained=True
                    )
                    model.eval()
                    
                    transform = T.Compose([
                        T.Resize((448, 448)),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    
                    # Load tags if specified
                    tags = None
                    if "selected_tags.csv" in model_config["files"]:
                        tags = await cls._load_wd14_tags(model_config["hf_path"])
                    
                    cls._models[model_config["model_id"]] = {
                        "model": model,
                        "transform": transform,
                        "tags": tags,
                        "type": "classification"
                    }
                    
                else:
                    # Generic classification model initialization
                    model = timm.create_model(
                        f"hf_hub:{model_config['hf_path']}", 
                        pretrained=True
                    )
                    model.eval()
                    
                    cls._models[model_config["model_id"]] = {
                        "model": model,
                        "type": "classification"
                    }
                    
                logger.info(f"Initialized classification model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
                raise

    @staticmethod
    async def _load_wd14_tags(model_path: str) -> list:
        """Load WD14 tags from the tags file."""
        import csv
        from huggingface_hub import hf_hub_download
        
        try:
            tags_path = hf_hub_download(
                repo_id=model_path,
                filename="selected_tags.csv",
                cache_dir=HF_CACHE_DIR
            )
            
            with open(tags_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                return [row[0] for row in reader]
                
        except Exception as e:
            logger.error(f"Failed to load WD14 tags: {e}")
            raise

    @classmethod
    def get_model(cls, model_id: str) -> Dict[str, Any]:
        """
        Get an initialized model by its ID.
        
        Args:
            model_id: The model's identifier from its config
            
        Returns:
            Dict containing the model and its associated data
            
        Raises:
            RuntimeError: If models aren't initialized
            KeyError: If model_id isn't found
        """
        if not cls._initialized:
            raise RuntimeError("Models not initialized. Call initialize() first.")
            
        model_info = cls._models.get(model_id)
        if model_info is None:
            raise KeyError(f"Model {model_id} not found in registry")
            
        return model_info

    @classmethod
    async def cleanup(cls) -> None:
        """Clean up model resources."""
        try:
            for model_info in cls._models.values():
                model = model_info.get("model")
                if model is not None:
                    del model
            
            cls._models.clear()
            cls._initialized = False
            torch.cuda.empty_cache()  # Clear CUDA cache if used
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cls._initialized = False