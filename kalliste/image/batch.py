from pathlib import Path
from typing import List
from .original_image import OriginalImage
from .types import ProcessingStatus
import asyncio

class Batch:
    def __init__(self, batch_path: Path):
        self.path = batch_path
        self.original_images: List[OriginalImage] = []
        self.status = ProcessingStatus.PENDING
        self.scan_for_images()
        
    def scan_for_images(self):
        """Find all images in batch folder"""
        for file in self.path.glob("*.jpg"):  # Add other extensions as needed
            self.original_images.append(OriginalImage(file))
            
    async def process(self):
        """Process all images in batch asynchronously"""
        self.status = ProcessingStatus.PROCESSING
        
        try:
            tasks = [image.process() for image in self.original_images]
            results = await asyncio.gather(*tasks)
            
            self.status = ProcessingStatus.COMPLETE
            # Signal completion to batch processor
            await self.on_complete()
            return results
            
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            print(f"Error processing batch {self.path}: {e}")
            raise
        
    async def on_complete(self):
        """Called when all images in batch are complete"""
        print(f"Batch {self.path} complete")
        
    async def on_image_complete(self, image_path: Path):
        """Called when an original image signals completion"""
        print(f"Image {image_path} in batch {self.path} complete")