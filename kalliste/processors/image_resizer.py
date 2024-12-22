from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image

class ImageResizer:
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        self.target_size = target_size
        
    def resize_image(self,
                    image: Union[str, Path, Image.Image],
                    output_path: Optional[Union[str, Path]] = None,
                    size: Optional[Tuple[int, int]] = None,
                    maintain_aspect: bool = True) -> Tuple[int, int]:
        """Resize image while optionally maintaining aspect ratio."""
        if size is None:
            size = self.target_size
        if size is None:
            raise ValueError("No target size specified")
            
        # Handle input type
        if isinstance(image, Image.Image):
            img = image
            needs_close = False
        else:
            img = Image.open(Path(image))
            needs_close = True
        
        try:
            if maintain_aspect:
                # Calculate new dimensions maintaining aspect ratio
                orig_width, orig_height = img.size
                target_width, target_height = size
                
                # Calculate scaling factors and use smaller ratio
                width_ratio = target_width / orig_width
                height_ratio = target_height / orig_height
                scale = min(width_ratio, height_ratio)
                
                # Calculate new dimensions
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save resized image if output path provided
            if output_path is not None:
                output_path = Path(output_path)
                resized.save(output_path, quality=95, optimize=True)
            
            return resized.size
            
        finally:
            if needs_close:
                img.close()
            
    def resize_and_pad(self,
                      image: Union[str, Path, Image.Image],
                      output_path: Optional[Union[str, Path]] = None,
                      size: Optional[Tuple[int, int]] = None,
                      background_color: Union[Tuple[int, int, int], str] = (255, 255, 255)) -> Tuple[int, int]:
        """Resize image maintaining aspect ratio and pad to target size."""
        if size is None:
            size = self.target_size
        if size is None:
            raise ValueError("No target size specified")
        
        # Handle input type
        if isinstance(image, Image.Image):
            img = image
            needs_close = False
        else:
            img = Image.open(Path(image))
            needs_close = True
            
        try:
            target_width, target_height = size
            
            # Calculate dimensions maintaining aspect ratio
            orig_width, orig_height = img.size
            width_ratio = target_width / orig_width
            height_ratio = target_height / orig_height
            scale = min(width_ratio, height_ratio)
            
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create padded image
            result = Image.new("RGB", size, background_color)
            
            # Calculate padding
            left = (target_width - new_width) // 2
            top = (target_height - new_height) // 2
            
            # Paste resized image onto padded background
            result.paste(resized, (left, top))
            
            # Save result if output path provided
            if output_path is not None:
                output_path = Path(output_path)
                result.save(output_path, quality=95, optimize=True)
            
            return result.size
            
        finally:
            if needs_close:
                img.close()