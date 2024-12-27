import asyncio
import logging
from typing import Any
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
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
                # First verify/download models
                downloader = ModelDownloadManager()
                await downloader.download_all()
                
                # Now load from cache (HF will find them automatically)
                model_name = "SmilingWolf/wd-vit-large-tagger-v3"
                
                logger.info(f"Loading model from {model_name}")
                model = AutoModelForImageClassification.from_pretrained(model_name)
                if model is None:
                    raise RuntimeError("Failed to load model")
                    
                logger.info(f"Loading feature extractor from {model_name}")
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                if feature_extractor is None:
                    raise RuntimeError("Failed to load feature extractor")
                
                cls._models['wd14'] = {
                    'model': model,
                    'feature_extractor': feature_extractor
                }
                
                logger.info("Models loaded successfully")
                cls._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize models: {e}")
                await cls.cleanup()
                raise

    @classmethod
    def get_model(cls, model_name: str) -> Any:
        if not cls._initialized:
            raise RuntimeError("Models not initialized")
        return cls._models.get(model_name)

    @classmethod
    async def cleanup(cls):
        cls._models.clear()
        cls._initialized = False