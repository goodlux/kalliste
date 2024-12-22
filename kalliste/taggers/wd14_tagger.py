"""WD14 image tagging implementation."""

from typing import Dict, List, Union, Any, Optional, Set
from pathlib import Path
import pandas as pd
import torch
import timm
import logging
import torchvision.transforms as T
from PIL import Image

from .base_tagger import BaseTagger, TagResult
from ..config import WD14_MODEL_ID, WD14_TAGS_FILE, WD14_WEIGHTS_DIR

logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """Image tagging using WD14 model."""
    
    # Default tags to exclude
    DEFAULT_BLACKLIST = {
        'questionable', 'explicit', 'nude', 'nudity', 'nipples', 'pussy',
        'penis', 'sex', 'cum', 'penetration', 'penetrated'
    }
    
    def __init__(
        self,
        model_id: str = WD14_MODEL_ID,
        tags_file: str = WD14_TAGS_FILE,
        blacklist: Optional[Set[str]] = None,
        confidence_threshold: float = 0.35,
        category_filters: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize WD14 tagger.
        
        Args:
            model_id: Model identifier for timm
            tags_file: Path to CSV file containing tag definitions
            blacklist: Set of tags to exclude (defaults to DEFAULT_BLACKLIST)
            confidence_threshold: Minimum confidence for tag inclusion
            category_filters: Optional list of categories to include
            **kwargs: Additional arguments passed to BaseTagger
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.tags_file = Path(tags_file)
        self.blacklist = blacklist or self.DEFAULT_BLACKLIST
        self.threshold = confidence_threshold
        self.category_filters = set(category_filters) if category_filters else None
        
        # Ensure weights directory exists
        WD14_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.transform = None
        self.tags_df = None
        self._load_model()
    
    def _load_model(self):
        """Load WD14 model and tag definitions."""
        try:
            # Load model
            logger.info(f"Loading WD14 model: {self.model_id}")
            self.model = timm.create_model(
                self.model_id,
                pretrained=True
            ).to('cpu')  # Always use CPU for WD14
            self.model.eval()
            
            # Load tag definitions
            if not self.tags_file.exists():
                raise FileNotFoundError(f"Tags file not found: {self.tags_file}")
            
            self.tags_df = pd.read_csv(self.tags_file)
            logger.info(f"Loaded {len(self.tags_df)} WD14 tags")
            
            # Set up image transforms
            self.transform = T.Compose([
                T.Resize((448, 448)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Failed to load WD14 model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for WD14 model."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to('cpu')
    
    def _filter_tags(self, tag: str, confidence: float, category: str) -> bool:
        """Check if tag should be included based on filters."""
        if confidence < self.threshold:
            return False
        if tag in self.blacklist:
            return False
        if self.category_filters and category not in self.category_filters:
            return False
        return True
    
    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, List[TagResult]]:
        """Convert model output to TagResults."""
        # Get probabilities
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
        
        # Create tag results
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
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return {'wd14': results}
    
    async def tag_image(self, image_path: Union[str, Path]) -> Dict[str, List[TagResult]]:
        """Generate tags for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with 'wd14' key mapping to list of TagResults
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self._preprocess_image(image)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Process results
            return self._postprocess_output(output)
            
        except Exception as e:
            logger.error(f"WD14 tagging failed: {e}")
            return {'wd14': []}