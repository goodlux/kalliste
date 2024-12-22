"""Test script for tagger pipeline."""

import asyncio
from pathlib import Path
import logging
from typing import Optional

from kalliste.taggers.tagger_pipeline import TaggerPipeline
from kalliste.taggers.config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pipeline(
    image_path: Path,
    config: Optional[PipelineConfig] = None
) -> None:
    """Test the tagger pipeline on an image."""
    
    # Use default config if none provided
    config = config or PipelineConfig()
    
    # Initialize pipeline
    pipeline = TaggerPipeline(config)
    
    try:
        # Process image
        results = await pipeline.tag_image(image_path)
        
        # Print results
        logger.info("Tagging Results:")
        for category, tags in results.items():
            logger.info(f"\n{category.upper()}:")
            for tag in tags:
                logger.info(f"  {tag}")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

async def main():
    logger.info("Starting pipeline test...")
    
    # Example configuration
    config = PipelineConfig(
        wd14_confidence=0.4,  # Slightly higher confidence threshold
        wd14_categories=['character', 'style']  # Only include these categories
    )
    logger.info("Configuration loaded")
    
    # Test directory
    test_dir = Path("/Users/rob/repos/kalliste/test_images/02_test/input")
    logger.info(f"Looking for images in: {test_dir}")
    
    # Process all PNG images in test directory
    image_paths = list(test_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} PNG files")
    
    for image_path in image_paths:
        logger.info(f"\nProcessing {image_path}")
        await test_pipeline(image_path, config)

if __name__ == "__main__":
    logger.info("Script starting...")
    asyncio.run(main())
    logger.info("Script finished.")