"""WD14 image tagging implementation."""

from typing import Dict, List, Union, Any, Optional, Set,  Dict, List
from pathlib import Path
import pandas as pd
import torch
import logging
import torchvision.transforms as T
from PIL import Image

from .base_tagger import BaseTagger
from ..types import TagResult
from ..model.model_registry import ModelRegistry

from pathlib import Path
from .base_tagger import BaseTagger

 



logger = logging.getLogger(__name__)

class WD14Tagger(BaseTagger):
    """WD14-based image tagger for general content tagging."""
    
    def __init__(self, confidence_threshold: float = 0.35, blacklist: List[str] = None):
        """Initialize WD14 tagger with config."""
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.blacklist = set(blacklist or [])
        self.model = None
        
    async def tag_image(self, image_path: Path) -> Dict[str, List[TagResult]]:
        """Generate WD14 tags for an image."""
        if self.model is None:
            self.model = ModelRegistry.get_tagger('wd14')
            
        try:
            # ... existing processing code ...
            
            # Filter results based on config
            filtered_results = []
            for tag, confidence in zip(tags, confidences):
                if (confidence >= self.confidence_threshold and 
                    tag not in self.blacklist):
                    filtered_results.append(
                        TagResult(
                            label=tag,
                            confidence=confidence,
                            category='general'
                        )
                    )
            
            return {'wd14': filtered_results}
            
        except Exception as e:
            logger.error(f"WD14 tagging failed: {e}", exc_info=True)
            raise