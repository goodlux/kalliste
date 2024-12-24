import asyncio
import logging
from typing import Optional
from ..detectors.base import DetectionConfig
from ..detectors.yolo_detector import YOLODetector
from ..taggers.tagger_pipeline import TaggerPipeline

logger = logging.getLogger(__name__)

class ModelRegistry:
    _models = {}
    _initialized = False
    _lock = asyncio.Lock()  # For thread-safe initialization

    @classmethod
    async def initialize(cls):
        """Initialize all models"""
        logger.info("Starting model registry initialization")
        async with cls._lock:  # Prevent concurrent initialization
            if cls._initialized:
                logger.info("Model registry already initialized")
                return
                
            try:
                logger.info("Initializing tagger pipeline...")
                # Create and initialize tagger pipeline
                tagger = TaggerPipeline()
                await tagger.initialize()
                cls._models['tagger'] = tagger
                logger.info("Tagger pipeline initialized successfully")

                logger.info("Initializing YOLO detector...")
                # Initialize detector with configs for each detection type
                detector_configs = [
                    DetectionConfig(
                        name='face',
                        confidence_threshold=0.5,
                        preferred_aspect_ratios=[(1, 1)]  # Square for faces
                    ),
                    DetectionConfig(
                        name='person',
                        confidence_threshold=0.5,
                        preferred_aspect_ratios=[(3, 4), (4, 5)]  # Portrait for people
                    )
                ]
                cls._models['detector'] = YOLODetector(config=detector_configs)
                logger.info("YOLO detector initialized successfully")

                cls._initialized = True
                logger.info("Model registry initialization complete")

            except Exception as e:
                logger.error(f"Failed to initialize models: {e}", exc_info=True)
                # Cleanup any partially initialized models
                await cls.cleanup()
                raise RuntimeError(f"Failed to initialize models: {e}")

    @classmethod
    async def cleanup(cls):
        """Cleanup all models"""
        logger.info("Starting model registry cleanup")
        async with cls._lock:
            for name, model in cls._models.items():
                try:
                    logger.debug(f"Cleaning up {name}...")
                    if hasattr(model, 'cleanup'):
                        await model.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}", exc_info=True)
            
            cls._models.clear()
            cls._initialized = False
            logger.info("Model registry cleanup complete")

    @classmethod
    def get_tagger(cls) -> TaggerPipeline:
        """Get the tagger pipeline instance"""
        if not cls._initialized:
            logger.error("Attempted to get tagger before initialization")
            raise RuntimeError("Models not initialized")
        logger.debug("Retrieved tagger from registry")
        return cls._models['tagger']

    @classmethod
    def get_detector(cls) -> YOLODetector:
        """Get the detector instance"""
        if not cls._initialized:
            logger.error("Attempted to get detector before initialization")
            raise RuntimeError("Models not initialized")
        logger.debug("Retrieved detector from registry")
        return cls._models['detector']