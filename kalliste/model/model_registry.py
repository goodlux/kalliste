import asyncio
import logging
from typing import Any
import timm
import torch
import torchvision.transforms as T
from .model_download_manager import ModelDownloadManager

logger = logging.getLogger(__name__)

class ModelRegistry:
    _models = {}
    _initialized = False
    _lock = asyncio.Lock()

    @classmethod
    async def initialize(cls):
        """Initialize all models upfront"""
        async with cls._lock:
            if cls._initialized:
                return
                
            try:
                # First verify/download using ModelDownloadManager
                downloader = ModelDownloadManager()
                tag_info = await downloader.load_tags()
                
                # Load model using timm
                model_name = "hf_hub:SmilingWolf/wd-vit-large-tagger-v3"
                logger.info(f"Loading model from {model_name}")
                
                model = timm.create_model(model_name, pretrained=True)
                model.eval()
                
                # Set up transforms 
                transform = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                ])
                
                cls._models['wd14'] = {
                    'model': model,
                    'transform': transform,
                    'tags': tag_info
                }
                
                logger.info("Model loaded successfully")
                cls._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize models: {e}", exc_info=True)
                await cls.cleanup()
                raise

    @classmethod
    def get_tagger(cls, model_name: str) -> Any:
        """Get initialized tagger by name."""
        if not cls._initialized:
            raise RuntimeError("Models not initialized")
            
        model_info = cls._models.get(model_name)
        if model_info is None:
            raise KeyError(f"Model {model_name} not found in registry")
            
        return model_info

    @classmethod
    async def cleanup(cls):
        """Clean up model resources"""
        try:
            for model_info in cls._models.values():
                model = model_info.get('model')
                if model is not None:
                    del model
            
            cls._models.clear()
            cls._initialized = False
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cls._initialized = False