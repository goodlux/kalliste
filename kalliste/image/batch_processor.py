from pathlib import Path
from typing import List, Optional, Dict
from .batch import Batch
from ..types import ProcessingStatus
from ..model.model_registry import ModelRegistry
import asyncio
import signal
import logging
import yaml

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process batches of images, where each batch is a directory in the input path."""
    
    def __init__(self, input_path: str, output_path: str, config: Optional[Dict] = None):
        """
        Initialize batch processor with input and output paths.
        
        Args:
            input_path: String path to root directory containing batch folders
            output_path: String path to root directory for processed output
            config: Optional processing configuration
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config = config or self._load_config()
        self.batches: List[Batch] = []
        self.status = ProcessingStatus.PENDING
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet, will set up in process_all
            return
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._handle_shutdown(s)))

    async def _handle_shutdown(self, sig):
        """Handle shutdown signal gracefully."""
        logger.info(f"Received signal {sig.name}, initiating shutdown...")
        self._shutdown_event.set()
        
        # Cancel any running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup
        try:
            logger.info("Cleaning up model registry...")
            await ModelRegistry.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        logger.info("Shutdown complete")

    def _load_config(self) -> Dict:
        """Load default configuration file."""
        config_path = Path(__file__).parent.parent / 'default_detection_config.yaml'
        try:
            return yaml.safe_load(config_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}", exc_info=True)
            raise RuntimeError(f"Cannot load configuration: {e}") from e

    async def setup(self) -> None:
        """Initialize model registry and discover batches."""
        logger.info(f"Setting up processor - Input: {self.input_path}, Output: {self.output_path}")
        
        # Initialize model registry
        await ModelRegistry.initialize()
        
        # Find and create batches
        self.batches = []
        for folder in self.input_path.iterdir():
            if not folder.is_dir():
                continue
                
            # Create corresponding output directory
            batch_output = self.output_path / folder.name
            batch_output.mkdir(parents=True, exist_ok=True)
            
            # Create batch
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
        """Process all batches in the input directory."""
        if not self.batches:
            await self.setup()
            if not self.batches:
                logger.warning("No batches found to process")
                return

        # Ensure signal handlers are set up in this loop
        self._setup_signal_handlers()

        logger.info("Starting batch processing")
        self.status = ProcessingStatus.PROCESSING
        
        # Count total images
        total_images = sum(len(batch.images) for batch in self.batches)
        processed_images = 0
        errors = []

        try:
            # Process each batch
            for batch in self.batches:
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested, stopping processing")
                    break
                    
                try:
                    await batch.process()
                    processed_images += len(batch.images)
                    logger.info(f"Progress: {processed_images}/{total_images} images")
                except Exception as e:
                    errors.append((batch.input_path.name, str(e)))
                    logger.error(f"Failed to process {batch.input_path.name}", exc_info=True)

            # Handle any errors
            if errors:
                error_msg = "\n".join(f"Batch {name}: {error}" for name, error in errors)
                raise RuntimeError(f"Failed to process {len(errors)} batch(es):\n{error_msg}")

            if not self._shutdown_event.is_set():
                self.status = ProcessingStatus.COMPLETE
                logger.info(f"Processing complete: {total_images} images in {len(self.batches)} batches")
            else:
                self.status = ProcessingStatus.ERROR
                logger.info("Processing interrupted by user")

        except asyncio.CancelledError:
            self.status = ProcessingStatus.ERROR
            logger.info("Processing cancelled")
            raise
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            raise
            
        finally:
            if not self._shutdown_event.is_set():
                try:
                    await ModelRegistry.cleanup()
                except Exception as e:
                    logger.error("Failed to cleanup model registry", exc_info=True)