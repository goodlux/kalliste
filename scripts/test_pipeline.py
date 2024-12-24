#!/usr/bin/env python3

import sys
import os
import asyncio
import logging
from pathlib import Path
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kalliste.image.batch_processor import BatchProcessor
from kalliste.utils import format_error

# Install rich traceback handling
install_rich_traceback(show_locals=True, width=None, word_wrap=True)

# Set up logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Rich handler adds its own formatting
    handlers=[
        RichHandler(rich_tracebacks=True, markup=True),  # Console handler with rich formatting
        logging.FileHandler('pipeline_test.log')  # Keep plain file handler for logs
    ]
)

logger = logging.getLogger(__name__)

async def main():
    # Set up test directories
    input_root = project_root / "test_images_input"
    output_root = project_root / "test_images_output"
    
    logger.info("[blue]Starting pipeline test[/]")
    logger.info(f"Input directory: [cyan]{input_root}[/]")
    logger.info(f"Output directory: [cyan]{output_root}[/]")
    
    # Make sure output directory exists
    output_root.mkdir(parents=True, exist_ok=True)
    logger.debug("Created output directory")
    
    # Initialize BatchProcessor with input and output roots
    try:
        logger.info("[green]Initializing BatchProcessor[/]")
        processor = BatchProcessor(
            input_root=input_root,
            output_root=output_root
        )
        logger.info("BatchProcessor created")
        
        logger.info("Initializing models")
        await processor.initialize()
        logger.info("[green]BatchProcessor initialized successfully[/]")
    except Exception as e:
        logger.error("[red bold]Failed to initialize BatchProcessor[/]")
        format_error(e, "Initialization Error")
        raise
    
    # Process all batches
    try:
        logger.info("[green]Starting batch processing[/]")
        await processor.process_all()
        logger.info("[green bold]All batches processed successfully![/]")
    except Exception as e:
        logger.error("[red bold]Error during batch processing[/]")
        format_error(e, "Processing Error")
        raise
    finally:
        logger.info("[blue]Pipeline test complete[/]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[yellow]Pipeline test interrupted by user[/]")
        sys.exit(1)
    except Exception as e:
        logger.error("[red bold]Pipeline test failed[/]")
        format_error(e, "Fatal Error")
        sys.exit(1)