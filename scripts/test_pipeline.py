#!/usr/bin/env python3
import asyncio
from kalliste.image.batch_processor import BatchProcessor
from kalliste import utils  # Sets up logging

async def main():
    processor = BatchProcessor(
        input_path='/Volumes/g2/kalliste_photos/kalliste_input',
        output_path='/Volumes/m01/kalliste_data/images'
    )
    await processor.setup()
    await processor.process_all()

if __name__ == "__main__":
    asyncio.run(main())