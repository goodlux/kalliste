"""
Model Download Manager for Kalliste

Handles downloading and verifying AI models from various sources.
Uses standard cache locations:
- YOLO models: ~/.cache/ultralytics
- HuggingFace models: ~/.cache/huggingface/hub
"""

import asyncio
import logging
from pathlib import Path
import aiohttp
import aiofiles
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .config import MODELS, YOLO_CACHE_DIR, HF_CACHE_DIR

logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Raised when a model download fails"""
    pass

class ModelDownloadManager:
    """
    Manages downloading and verifying models defined in config.MODELS.
    
    Downloads models to standard cache locations:
    - YOLO models: ~/.cache/ultralytics
    - HuggingFace models: ~/.cache/huggingface/hub
    
    Models are defined in config.MODELS with consistent structure:
    - Detection models have 'file' and 'url'
    - Classification models have 'hf_path' and 'files'
    """
    
    def __init__(self):
        """Initialize the download manager and ensure cache directories exist."""
        self._ensure_cache_dirs()
        
    def _ensure_cache_dirs(self) -> None:
        """Ensure all cache directories exist."""
        YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cache directories verified")
        
    async def download_all(self) -> None:
        """Download and verify all models defined in config."""
        try:
            logger.info("Starting model downloads")
            
            # Download detection models
            for model_name, model_config in MODELS["detection"].items():
                await self._download_detection_model(model_name, model_config)
                
            # Download classification models
            for model_name, model_config in MODELS["classification"].items():
                await self._download_classification_model(model_name, model_config)
                
            logger.info("All models downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            raise DownloadError(f"Model download failed: {str(e)}") from e
            
    async def _download_detection_model(self, name: str, config: dict) -> None:
        """Download a detection (YOLO) model if it doesn't exist."""
        model_path = YOLO_CACHE_DIR / config["file"]
        
        if model_path.exists():
            logger.info(f"Detection model already exists: {name}")
            return
            
        logger.info(f"Downloading detection model: {name}")
        try:
            # For YOLO models from custom URLs, download directly
            async with aiohttp.ClientSession() as session:
                async with session.get(config["url"]) as response:
                    if response.status != 200:
                        raise DownloadError(
                            f"Failed to download {name} from {config['url']}"
                        )
                    
                    async with aiofiles.open(model_path, 'wb') as f:
                        await f.write(await response.read())
                        
            # Verify the model loads correctly
            try:
                YOLO(str(model_path))
                logger.info(f"Successfully downloaded and verified {name}")
            except Exception as e:
                model_path.unlink(missing_ok=True)  # Clean up bad download
                raise DownloadError(f"Downloaded model {name} failed verification") from e
                
        except Exception as e:
            raise DownloadError(f"Error downloading {name}: {str(e)}") from e
            
    async def _download_classification_model(self, name: str, config: dict) -> None:
        """Download a classification model and its associated files from HuggingFace."""
        try:
            logger.info(f"Downloading classification model: {name}")
            
            # Download the model and any additional files
            hf_path = config["hf_path"]
            
            # First verify/download the main model
            hf_hub_download(repo_id=hf_path, cache_dir=HF_CACHE_DIR)
            
            # Then download any additional files
            for file in config["files"]:
                hf_hub_download(
                    repo_id=hf_path,
                    filename=file,
                    cache_dir=HF_CACHE_DIR
                )
                
            logger.info(f"Successfully downloaded {name} and all associated files")
            
        except Exception as e:
            raise DownloadError(f"Error downloading {name}: {str(e)}") from e
    
    @staticmethod
    def verify_model_exists(model_config: dict) -> bool:
        """
        Verify if a model exists in its cache location.
        
        Args:
            model_config: Model configuration dictionary from MODELS
            
        Returns:
            bool: True if model exists and is valid
        """
        try:
            if "file" in model_config:  # Detection model
                model_path = YOLO_CACHE_DIR / model_config["file"]
                return model_path.exists()
            else:  # Classification model
                # Check if we can access the model in cache
                hf_hub_download(repo_id=model_config["hf_path"], cache_dir=HF_CACHE_DIR)
                # Check additional files if any
                for file in model_config["files"]:
                    hf_hub_download(
                        repo_id=model_config["hf_path"],
                        filename=file,
                        cache_dir=HF_CACHE_DIR
                    )
                return True
        except Exception:
            return False