import asyncio
import logging
import httpx
from pathlib import Path
from huggingface_hub import snapshot_download
from ..config import MODELS, YOLO_CACHE_DIR, HF_CACHE_DIR, NIMA_CACHE_DIR
import os
# Convert string paths to expanded Path objects
YOLO_CACHE_DIR = Path(os.path.expanduser(YOLO_CACHE_DIR))
HF_CACHE_DIR = Path(os.path.expanduser(HF_CACHE_DIR))
NIMA_CACHE_DIR = Path(os.path.expanduser(NIMA_CACHE_DIR))

logger = logging.getLogger(__name__)

class ModelDownloadManager:
    """Downloads YOLO and HuggingFace models to their respective cache directories."""
    
    def __init__(self):
        """Initialize download manager and ensure cache directories exist."""
        YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        NIMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
    async def download_all(self):
        """Downloads all required models."""
        logger.info("Starting model downloads")
        try:
            # Download YOLO models
            await self._download_yolo_models()
            # Download HuggingFace models
            await self._download_huggingface_models()
            # Download NIMA models
            await self._download_nima_models()
            # Download embedding models
            await self._download_embedding_models() 
            logger.info("All model downloads complete")
            
        except Exception as e:
            logger.error(f"Failed to download models: {e}", exc_info=True)
            raise RuntimeError(f"Model download failed: {str(e)}") from e


    async def _download_embedding_models(self):
        """Download embedding models from HuggingFace hub."""
        exclude_patterns = [
            "*.onnx",
            "*.msgpack",
            "flax_model.*",
            "tf_model.*",
            "*.md",
            ".git*",
            "*.txt",
            "*.bin",
            "pytorch_model.*",
            "*.h5",
        ]
        
        for name, config in MODELS["embeddings"].items():
            logger.info(f"Checking/downloading {name} from HuggingFace")
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=config["hf_path"],
                    cache_dir=HF_CACHE_DIR,
                    ignore_patterns=exclude_patterns,
                    local_files_only=False,
                    resume_download=True
                )
            )
            logger.info(f"Successfully downloaded {name}")


    async def _download_yolo_models(self):
        """Download YOLO detection models."""
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

    async def _download_huggingface_models(self):
        """Download models from HuggingFace hub."""
        # Define patterns to exclude from downloads
        exclude_patterns = [
            "*.onnx",          # ONNX models
            "*.msgpack",       # JAX/Flax serialized models
            "flax_model.*",    # Flax model files
            "tf_model.*",      # TensorFlow models
            "*.md",            # Documentation
            ".git*",           # Git files
            "*.txt",           # Text files
            "*.bin",           # Prefer safetensors over bin files
            "pytorch_model.*", # Generic PyTorch files
            "*.h5",           # HDF5 files
        ]
        
        # Download HuggingFace models
        for name, config in MODELS["classification"].items():
            # Skip NIMA as it's not from HuggingFace
            if name == "nima":
                continue
                
            logger.info(f"Checking/downloading {name} from HuggingFace")
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=config["hf_path"],
                    cache_dir=HF_CACHE_DIR,
                    ignore_patterns=exclude_patterns,
                    local_files_only=False,  # Force check for updates
                    resume_download=True     # Resume partial downloads
                )
            )
            logger.info(f"Successfully downloaded {name}")

    async def _download_nima_models(self):
        """Download NIMA technical and aesthetic models."""
        nima_config = MODELS["classification"]["nima"]
        
        for model_type in ["technical", "aesthetic"]:
            filepath = NIMA_CACHE_DIR / nima_config["files"][model_type]
            url = nima_config["urls"][model_type]
            
            if filepath.exists() and filepath.stat().st_size > 0:
                logger.info(f"NIMA {model_type} model already exists at {filepath}")
                continue
                
            logger.info(f"Downloading NIMA {model_type} model from {url}")
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                filepath.write_bytes(response.content)
            logger.info(f"Successfully downloaded NIMA {model_type} model")