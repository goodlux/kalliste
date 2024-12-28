#!/usr/bin/env python3

import sys
import os
import asyncio
import logging
from pathlib import Path
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kalliste.image.batch_processor import BatchProcessor
from kalliste.utils import format_error

logger = logging.getLogger(__name__)

async def main():
    logger.info("[blue]Starting pipeline test[/]")
    
    # Load default config
    config_path = project_root / 'kalliste' / 'default_detection_config.yaml'
    logger.info(f"Loading config from: [cyan]{config_path}[/]")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config", exc_info=True)
        raise
        
    logger.info(f"Input directory: [cyan]{project_root / 'test_images_input'}[/]")
    logger.info(f"Output directory: [cyan]{project_root / 'test_images_output'}[/]")
    
    processor = None
    
    try:
        logger.info("[green]Initializing BatchProcessor[/]")
        input_root = project_root / 'test_images_input'
        output_root = project_root / 'test_images_output'
        
        # Pass config to BatchProcessor
        processor = BatchProcessor(
            input_root=input_root, 
            output_root=output_root,
            config=config
        )
        await processor.initialize()
        await processor.process_all()
    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        raise
    finally:
        if processor:
            await processor.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)