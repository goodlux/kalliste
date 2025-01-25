import asyncio
import logging
import timm
import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
from typing import Dict, Any

from ..config import MODELS, YOLO_CACHE_DIR, NIMA_CACHE_DIR
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
                # Determine device once
                if torch.cuda.is_available():
                    cls.device = 'cuda'
                elif torch.backends.mps.is_available():
                    cls.device = 'mps'
                else:
                    cls.device = 'cpu'
                logger.info(f"Using device: {cls.device}")
                
                # Configure TensorFlow GPU memory growth
                if cls.device == 'cuda':
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                
                # First ensure all models are downloaded
                downloader = ModelDownloadManager()
                await downloader.download_all()
                
                # Initialize models by type
                await cls._initialize_detection_models()
                await cls._initialize_captioning_models()      # BLIP2
                await cls._initialize_orientation_models()     # Orientation
                await cls._initialize_wd14_model()
                await cls._initialize_nima_models()           # NIMA aesthetic and technical

                cls._initialized = True
                logger.info("Model registry initialization complete")
                
            except Exception as e:
                logger.error(f"Failed to initialize model registry: {e}")
                await cls.cleanup()
                raise

    @classmethod
    async def _initialize_nima_models(cls) -> None:
        """Initialize NIMA technical and aesthetic quality models."""
        try:
            logger.info("Starting NIMA initialization")
            nima_config = MODELS["classification"]["nima"]
            
            def create_nima_model():
                """Create NIMA model architecture."""
                base_model = keras.applications.MobileNet(
                    input_shape=(224, 224, 3), 
                    include_top=False, 
                    pooling='avg'
                )
                
                x = keras.layers.Dropout(0.75)(base_model.output)
                x = keras.layers.Dense(10, activation='softmax')(x)
                
                return keras.models.Model(base_model.input, x)
            
            def create_nima_processor():
                """Create preprocessing function for NIMA."""
                def process_image(image):
                    """Process a PIL image for NIMA model input."""
                    # Convert PIL image to numpy array
                    img_array = np.array(image)
                    
                    # Ensure RGB
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]  # Remove alpha channel if present
                    
                    # Resize
                    img_array = tf.image.resize(img_array, (224, 224))
                    
                    # Preprocess for MobileNet
                    img_array = keras.applications.mobilenet.preprocess_input(img_array)
                    
                    # Add batch dimension if needed
                    if len(img_array.shape) == 3:
                        img_array = np.expand_dims(img_array, axis=0)
                    
                    return {"pixel_values": img_array}
                
                return process_image
            
            # Initialize both technical and aesthetic models
            for model_type in ["technical", "aesthetic"]:
                model = create_nima_model()
                weights_path = NIMA_CACHE_DIR / nima_config["files"][model_type]
                model.load_weights(str(weights_path))
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                
                # Store in registry with processor
                cls._models[f"nima_{model_type}"] = {
                    "model": model,
                    "processor": create_nima_processor(),
                    "type": "quality_assessment"
                }
                
                logger.info(f"Initialized NIMA {model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to initialize NIMA models: {e}", exc_info=True)
            raise

    @classmethod
    async def _initialize_detection_models(cls) -> None:
        """Initialize YOLO detection models."""
        for model_name, model_config in MODELS["detection"].items():
            try:
                model_path = YOLO_CACHE_DIR / model_config["file"]
                model = YOLO(str(model_path), task='detect')
                model.to(cls.device)
                
                cls._models[model_config["model_id"]] = {
                    "model": model,
                    "type": "detection"
                }
                logger.info(f"Initialized detection model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize detection model {model_name}: {e}")
                raise

    @classmethod
    async def _initialize_wd14_model(cls) -> None:
        """Initialize WD14 model using timm and standardized registry storage."""
        from huggingface_hub import hf_hub_download
        import pandas as pd
        import timm
        import torchvision.transforms as T
        
        try:
            logger.info("Starting WD14 initialization")
            model_config = MODELS["classification"]["wd14"]
            
            # Load tags and create mappings
            tags_path = hf_hub_download(
                repo_id=model_config['hf_path'],
                filename="selected_tags.csv"
            )
            tags_df = pd.read_csv(tags_path)
            id2label = {i: str(row['name']) for i, row in tags_df.iterrows()}
            id2category = {i: str(row['category']) for i, row in tags_df.iterrows()}
            
            # Initialize model with timm
            model = timm.create_model(
                f"hf_hub:{model_config['hf_path']}", 
                pretrained=True
            )
            model = model.to(cls.device)
            model.eval()
            
            # Create processor function to match other models' interface
            def processor(images, return_tensors="pt"):
                """Process images to match WD14 model requirements."""
                if not isinstance(images, list):
                    images = [images]
                    
                transform = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
                ])
                
                processed = [transform(img) for img in images]
                batch = torch.stack(processed).to(cls.device)
                
                return {"pixel_values": batch}
            
            # Store in registry with same structure as other models
            cls._models[model_config["model_id"]] = {
                "model": model,
                "processor": processor,
                "id2label": id2label,
                "id2category": id2category,
                "type": "classification"
            }
            
            logger.info(f"Initialized WD14 model with {len(id2label)} tags")
            
        except Exception as e:
            logger.error(f"Failed to initialize WD14 model: {e}", exc_info=True)
            raise

    @classmethod
    async def _initialize_huggingface_models(cls) -> None:
        """Initialize BLIP2 and orientation models using HuggingFace auto classes."""
        from transformers import AutoModelForImageClassification, AutoModelForConditionalGeneration
        from transformers import AutoProcessor

        model_configs = {
            "blip2": {
                "model_class": AutoModelForConditionalGeneration,
                "model_args": {
                    "torch_dtype": torch.float16 if cls.device == "cuda" else torch.float32
                }
            },
            "orientation": {
                "model_class": AutoModelForImageClassification,
                "model_args": {}
            }
        }

        for model_name, config in MODELS["classification"].items():
            # Skip models that need special handling
            if model_name in ["wd14", "nima"] or model_name not in model_configs:
                continue

            try:
                model_info = model_configs[model_name]
                
                model = model_info["model_class"].from_pretrained(
                    config['hf_path'],
                    **model_info["model_args"]
                ).to(cls.device)
                processor = AutoProcessor.from_pretrained(config['hf_path'])
                model.eval()

                cls._models[config["model_id"]] = {
                    "model": model,
                    "processor": processor,
                    "type": model_name
                }
                logger.info(f"Initialized {model_name} model from {config['hf_path']}")

            except Exception as e:
                logger.error(f"Failed to initialize {model_name} model: {e}")
                raise

    @classmethod
    async def _initialize_captioning_models(cls) -> None:
        """Initialize BLIP2 and other captioning/generation models."""
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        
        for model_name, model_config in MODELS["classification"].items():
            try:
                if model_name != "blip2":
                    continue  # Skip non-BLIP2 models
                    
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_config['hf_path'],
                    torch_dtype=torch.float16 if cls.device == "cuda" else torch.float32,
                ).to(cls.device)
                processor = Blip2Processor.from_pretrained(model_config['hf_path'])
                model.eval()
                
                cls._models[model_config["model_id"]] = {
                    "model": model,
                    "processor": processor,
                    "type": "captioning"
                }
                logger.info(f"Initialized captioning model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize captioning model {model_name}: {e}")
                raise

    @classmethod
    async def _initialize_orientation_models(cls) -> None:
        """Initialize orientation detection models."""
        from transformers import AutoModelForImageClassification, AutoFeatureExtractor
        
        try:
            for model_name, model_config in MODELS["classification"].items():
                if model_name != "orientation":
                    continue  # Skip non-orientation models
                    
                model = AutoModelForImageClassification.from_pretrained(
                    model_config['hf_path']
                ).to(cls.device)
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_config['hf_path'])
                model.eval()
                
                cls._models[model_config["model_id"]] = {
                    "model": model,
                    "processor": feature_extractor,  # Now matches what taggers expect
                    "type": "orientation"
                }
                logger.info(f"Initialized orientation model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize orientation model {model_name}: {e}")
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