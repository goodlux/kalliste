import asyncio
import logging
import httpx
from pathlib import Path
from huggingface_hub import snapshot_download
from ..config import MODELS, YOLO_CACHE_DIR, HF_CACHE_DIR

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    """Downloads YOLO and HuggingFace models to their respective cache directories."""
    
    def __init__(self):
        YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
    async def download_all(self):
        """Downloads all required models."""
        logger.info("Starting model downloads")
        try:
            # Download YOLO models
            for name, config in MODELS["detection"].items():
                filepath = YOLO_CACHE_DIR / config["file"]
                if filepath.exists() and filepath.stat().st_size > 0:
                    logger.info(f"Model {name} already exists at {filepath}")
                    continue
                    
                logger.info(f"Downloading {name} from {config['url']}")
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(config["url"])
                    response.raise_for_status()
                    filepath.write_bytes(response.content)
                logger.info(f"Successfully downloaded {name}")
                
            # Define patterns for HuggingFace downloads
            non_pytorch_patterns = [
                "*.onnx",          # ONNX models
                "*.msgpack",       # JAX/Flax serialized models
                "flax_model.*",    # Flax model files
                "*.safetensors",   # Alternative to PyTorch's .bin format
                "tf_model.*",      # TensorFlow models
                "*.md",            # Documentation
                ".git*",           # Git files
                "*.txt",           # Text files
            ]
            
            # Download HuggingFace models
            for name, config in MODELS["classification"].items():
                logger.info(f"Checking/downloading {name} from HuggingFace")
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: snapshot_download(
                        repo_id=config["hf_path"],
                        cache_dir=HF_CACHE_DIR,
                        ignore_patterns=non_pytorch_patterns  # Pass the patterns directly
                    )
                )
                logger.info(f"Successfully downloaded {name}")
                
            logger.info("All model downloads complete")
            
        except Exception as e:
            logger.error(f"Failed to download models: {e}", exc_info=True)
            raise RuntimeError(f"Model download failed: {str(e)}") from e