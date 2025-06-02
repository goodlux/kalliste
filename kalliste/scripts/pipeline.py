#!/usr/bin/env python3
"""
Pipeline entry point - modernized version of the test_pipeline script.
"""
import asyncio
from pathlib import Path
from kalliste.image.batch_processor import BatchProcessor
from kalliste import utils  # Sets up logging

def main():
    """Main entry point for the Kalliste pipeline."""
    asyncio.run(run_pipeline())

async def run_pipeline():
    """Run the full processing pipeline."""
    # Define paths - these should probably come from config eventually
    input_path = '/Volumes/g2/kalliste_photos/kalliste_input'
    output_path = '/Volumes/m01/kalliste_data/images'
    processed_path = '/Volumes/g2/kalliste_photos/kalliste_processed'
    
    # Create processed directory if it doesn't exist
    Path(processed_path).mkdir(exist_ok=True, parents=True)
    
    processor = BatchProcessor(
        input_path=input_path,
        output_path=output_path,
        processed_path=processed_path
    )
    await processor.setup()
    await processor.process_all()

if __name__ == "__main__":
    main()
