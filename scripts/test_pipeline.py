#!/usr/bin/env python3
import asyncio
from pathlib import Path
from kalliste.image.batch_processor import BatchProcessor
from kalliste import utils  # Sets up logging

async def main():
    # Define paths
    input_path = '/Volumes/m02/emily_test_1'
    output_path = '/Volumes/m01/kalliste_data/images'
    processed_path = '/Volumes/m02/emily_test_1_processed'

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
    asyncio.run(main())
