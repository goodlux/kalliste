from pathlib import Path
from typing import List, Optional
from .batch import Batch
from .types import ProcessingStatus
from .model_registry import ModelRegistry
import asyncio
import logging
import traceback

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, input_root: Path, output_root: Path):
        if not input_root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_root}")
            
        logger.info(f"Initializing BatchProcessor with input_root={input_root}, output_root={output_root}")
        self.input_root = input_root
        self.output_root = output_root
        self.batches: List[Batch] = []
        self.status = ProcessingStatus.PENDING
        self._initialized = False
        self.scan_for_batches()

    async def initialize(self):
        """Initialize models asynchronously"""
        if self._initialized:
            logger.debug("BatchProcessor already initialized")
            return

        logger.info("Initializing BatchProcessor and model registry")
        try:
            await ModelRegistry.initialize()
            self._initialized = True
            logger.info("BatchProcessor and model registry initialized successfully")
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error("Model registry initialization failed", exc_info=True)
            raise RuntimeError("Failed to initialize BatchProcessor") from e
            
    def create_output_directory(self, output_dir: Path) -> None:
        """Create output directory, handling errors gracefully"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}", exc_info=True)
            raise FileNotFoundError(f"Cannot create output directory {output_dir}") from e
        
    def scan_for_batches(self) -> None:
        """Scan input directory for batch folders"""
        logger.info(f"Scanning for batches in {self.input_root}")
        batch_count = 0
        
        try:
            for item in self.input_root.iterdir():
                if not item.is_dir():
                    continue
                    
                output_dir = self.output_root / item.name
                self.create_output_directory(output_dir)
                self.batches.append(Batch(input_path=item, output_path=output_dir))
                batch_count += 1
                logger.debug(f"Added batch: {item.name}")
                    
            if batch_count == 0:
                logger.warning(f"No batch directories found in {self.input_root}")
            else:
                logger.info(f"Found {batch_count} batches to process")
                
        except Exception as e:
            logger.error("Failed to scan for batches", exc_info=True)
            raise  # Preserve original error context
                
    async def process_batch(self, batch: Batch) -> Optional[Exception]:
        """Process a single batch, returning any error that occurred"""
        if not self._initialized:
            raise RuntimeError("Cannot process batch before initialization")

        logger.info(f"Starting batch: {batch.input_path.name}")
        try:
            await batch.process()
            logger.info(f"Completed batch: {batch.input_path.name}")
            return None
        except Exception as e:
            logger.error(f"Failed to process batch {batch.input_path.name}", exc_info=True)
            return e
                
    async def process_all(self):
        """Process batches sequentially, with improved error handling"""
        # Check initialization instead of forcing it
        if not self._initialized:
            logger.warning("BatchProcessor not initialized. Call initialize() first.")
            return
        
        logger.info("Starting batch processing")
        self.status = ProcessingStatus.PROCESSING
        
        if not self.batches:
            logger.warning("No batches to process")
            self.status = ProcessingStatus.COMPLETE
            return
        
        errors = []
        try:
            # Process batches one at a time
            for batch in self.batches:
                error = await self.process_batch(batch)
                if error:
                    errors.append((batch.input_path, error))
            
            # Handle any errors that occurred
            if errors:
                error_details = "\n".join(
                    f"Batch {path.name}: {str(err)}" 
                    for path, err in errors
                )
                raise RuntimeError(
                    f"Failed to process {len(errors)} batch(es):\n{error_details}"
                )
            
            self.status = ProcessingStatus.COMPLETE
            logger.info("All batches processed successfully")
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            logger.error("Batch processing failed", exc_info=True)
            raise  # Preserve original error context
        finally:
            # Always try to clean up
            try:
                logger.info("Cleaning up model registry")
                await ModelRegistry.cleanup()
                self._initialized = False
                logger.info("Model registry cleanup complete")
            except Exception as e:
                logger.error("Failed to clean up model registry", exc_info=True)
                # Don't raise cleanup errors - they're less important than processing errors

    async def on_batch_complete(self, batch_path: Path):
        """Called when a batch signals completion"""
        logger.info(f"Batch {batch_path} processing complete")