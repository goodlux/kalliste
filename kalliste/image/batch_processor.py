from pathlib import Path
from typing import List
from .batch import Batch
from .types import ProcessingStatus
import asyncio

class BatchProcessor:
    def __init__(self, root_data_path: Path):
        self.root_path = root_data_path
        self.batches: List[Batch] = []
        self.status = ProcessingStatus.PENDING
        self.scan_for_batches()
        
    def scan_for_batches(self):
        """Scan root directory for batch folders"""
        for item in self.root_path.iterdir():
            if item.is_dir():
                self.batches.append(Batch(item))
                
    async def process_all(self):
        """Process all batches asynchronously"""
        self.status = ProcessingStatus.PROCESSING
        
        try:
            # Create tasks for each batch
            tasks = [batch.process() for batch in self.batches]
            # Wait for all batches to complete
            results = await asyncio.gather(*tasks)
            
            self.status = ProcessingStatus.COMPLETE
            return results
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            print(f"Error processing batches: {e}")
            raise

    async def on_batch_complete(self, batch_path: Path):
        """Called when a batch signals completion"""
        print(f"Batch {batch_path} processing complete")