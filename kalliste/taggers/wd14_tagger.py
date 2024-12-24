"""WD14 image tagging implementation."""

from typing import Dict, List, Union, Any, Optional, Set
from pathlib import Path
import pandas as pd
import torch
import logging
import torchvision.transforms as T
from PIL import Image

from .base_tagger import BaseTagger
from ..image.types import TagResult

logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """WD14-based image tagger for general content tagging."""
    
    def __init__(
        self,
        model=None,
        tags_df=None,
        confidence_threshold: float = 0.35,
        category_filters: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize WD14Tagger with model components."""
        super().__init__(**kwargs)
        self.model = model
        self.tags_df = tags_df
        self.threshold = confidence_threshold
        self.category_filters = set(category_filters) if category_filters else None
        
        # Set up image transforms
        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225])
        ])
        
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for WD14 model."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to('cpu')  # WD14 always uses CPU
    
    def _filter_tags(self, tag: str, confidence: float, category: str) -> bool:
        """Check if tag should be included based on filters."""
        # TODO: Make more robust blocklist handling
        BLACKLIST = {
            'questionable', 'explicit', 'nude', 'nudity', 'nipples', 'pussy',
            'penis', 'sex', 'cum', 'penetration', 'penetrated'
        }
        
        if confidence < self.threshold:
            return False
        if tag in BLACKLIST:
            return False
        if self.category_filters and category not in self.category_filters:
            return False
        return True
    
    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, List[TagResult]]:
        """Convert model output to TagResults."""
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
        results = []
        
        for idx, (prob, tag_row) in enumerate(zip(probs, self.tags_df.itertuples())):
            tag = str(tag_row.name)
            category = str(tag_row.category)
            
            if self._filter_tags(tag, prob, category):
                results.append(
                    TagResult(
                        label=tag,
                        confidence=float(prob),
                        category=f'wd14_{category}'
                    )
                )
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        return {'wd14': results}
    
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate WD14 tags for an image."""
        if self.model is None or self.tags_df is None:
            raise RuntimeError("Model or tag data not provided. WD14Tagger requires initialized components.")
            
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            logger.debug("Image loaded successfully")
            
            img_tensor = self._preprocess_image(image)
            logger.debug("Image preprocessed successfully")

            with torch.no_grad():
                try:
                    output = self.model(img_tensor)
                    logger.debug("Model inference successful")
                except Exception as e:
                    logger.error(f"Error during model inference: {e}", exc_info=True)
                    raise
            
            return self._postprocess_output(output)
            
        except Exception as e:
            logger.error(f"WD14 tagging failed for {image_path}: {e}", exc_info=True)
            raise