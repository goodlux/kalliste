import asyncio
import logging
import httpx
from pathlib import Path
from huggingface_hub import snapshot_download

from ..config import MODELS, YOLO_CACHE_DIR, HF_CACHE_DIR

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    def __init__(self):
        self.config = MODELS
        
    async def download_all(self):
        """Downloads all models at startup"""
        logger.info("Starting model downloads")
        try:
            # Check and download detection models if needed
            for model_info in self.config["detection"].values():
                cache_path = YOLO_CACHE_DIR / model_info["file"]
                if not cache_path.exists():
                    logger.info(f"Downloading {model_info['model_id']} to {cache_path}")
                    async with httpx.AsyncClient() as client:
                        response = await client.get(model_info["url"])
                        cache_path.write_bytes(response.content)
                    logger.info(f"Successfully downloaded {model_info['model_id']}")
                else:
                    logger.info(f"Model {model_info['model_id']} already exists at {cache_path}")
                
            # Download HF models to cache
            non_pytorch_patterns = [
                "*.onnx",          # ONNX models
                "*.msgpack",       # JAX/Flax serialized models
                "flax_model.*",    # Flax model files
                "*.safetensors",   # Alternative to PyTorch's .bin format TODO: Change this to block .bin instead and use safetensors.
                "tf_model.*",      # TensorFlow models
                "*.md",            # Documentation
                ".git*",           # Git files
                "*.txt",           # Text files
            ]
            
            for model_info in self.config["classification"].values():
                logger.info(f"Checking/downloading {model_info['model_id']} from HuggingFace")
                try:
                    path = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=model_info["hf_path"],
                            ignore_patterns=non_pytorch_patterns,
                            cache_dir=HF_CACHE_DIR
                        )
                    )
                    logger.info(f"Successfully downloaded {model_info['model_id']} to {path}")
                except Exception as e:
                    logger.error(f"Failed to download {model_info['model_id']}: {str(e)}")
                    raise
                    
            logger.info("All model downloads complete")
            
        except Exception as e:
            logger.error(f"Failed to download models: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model download failed: {str(e)}") from e