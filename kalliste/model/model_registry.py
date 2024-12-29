import asyncio
import logging
import timm
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from typing import Dict, Any

from ..config import MODELS, YOLO_CACHE_DIR
from .model_download_manager import ModelDownloadManager

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton registry for initialized models.
    Models are initialized once and stored for reuse.
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
                    
                    cls._models[model_config["model_id"]] = {
                        "model": model,
                        "transform": transform,
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

    @classmethod
    def get_model(cls, model_id: str) -> Dict[str, Any]:
        """Get an initialized model by its ID."""
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
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cls._initialized = False