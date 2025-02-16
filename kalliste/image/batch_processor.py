from pathlib import Path
from typing import List, Optional, Dict
from .batch import Batch
from ..types import ProcessingStatus
from ..model.model_registry import ModelRegistry
import signal
import logging
import yaml
import sys

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, input_path: str, output_path: str, config: Optional[Dict] = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config = config or self._load_config()
        self.batches: List[Batch] = []
        self.status = ProcessingStatus.PENDING

    def _load_config(self) -> Dict:
        config_path = Path(__file__).parent.parent / 'default_detection_config.yaml'
        try:
            return yaml.safe_load(config_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}", exc_info=True)
            raise RuntimeError(f"Cannot load configuration: {e}") from e

    async def setup(self) -> None:
        logger.info(f"Setting up processor - Input: {self.input_path}, Output: {self.output_path}")
        
        # Initialize model registry
        await ModelRegistry.initialize()
        
        # Find and create batches
        self.batches = []
        for folder in self.input_path.iterdir():
            if not folder.is_dir():
                continue
                
            batch_output = self.output_path / folder.name
            batch_output.mkdir(parents=True, exist_ok=True)
            
            batch = Batch(
                input_path=folder,
                output_path=batch_output,
                default_config=self.config
            )
            self.batches.append(batch)
            logger.debug(f"Added batch: {folder.name}")
        
        n_batches = len(self.batches)
        logger.info(f"Found {n_batches} batch{'es' if n_batches != 1 else ''}")

    async def process_all(self) -> None:
        if not self.batches:
            await self.setup()
            if not self.batches:
                logger.warning("No batches found to process")
                return

        logger.info("Starting batch processing")
        self.status = ProcessingStatus.PROCESSING
        
        total_images = sum(len(batch.images) for batch in self.batches)
        processed_images = 0
        errors = []

        try:
            for batch in self.batches:
                try:
                    await batch.process()
                    processed_images += len(batch.images)
                    logger.info(f"Progress: {processed_images}/{total_images} images")
                except Exception as e:
                    errors.append((batch.input_path.name, str(e)))
                    logger.error(f"Failed to process {batch.input_path.name}", exc_info=True)

            if errors:
                error_msg = "\n".join(f"Batch {name}: {error}" for name, error in errors)
                raise RuntimeError(f"Failed to process {len(errors)} batch(es):\n{error_msg}")

            self.status = ProcessingStatus.COMPLETE
            logger.info(f"Processing complete: {total_images} images in {len(self.batches)} batches")

        except Exception as e:
            self.status = ProcessingStatus.ERROR
            raise
            
        finally:
            try:
                await ModelRegistry.cleanup()
            except Exception as e:
                logger.error("Failed to cleanup model registry", exc_info=True)