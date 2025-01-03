"""Image quality assessment using NIMA (Neural Image Assessment)."""
from typing import Dict, List, Optional
from PIL import Image
import logging
import numpy as np

from .AbstractTagger import AbstractTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class NIMATagger(AbstractTagger):
    """Tags images with aesthetic and technical quality scores using NIMA."""
    
    # Default configuration
    DEFAULT_TECHNICAL_THRESHOLD = 5.0  # Score from 1-10
    DEFAULT_AESTHETIC_THRESHOLD = 5.0  # Score from 1-10
    DEFAULT_USE_TECHNICAL = True
    DEFAULT_USE_AESTHETIC = True
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for NIMA tagger."""
        return {
            'technical_threshold': cls.DEFAULT_TECHNICAL_THRESHOLD,
            'aesthetic_threshold': cls.DEFAULT_AESTHETIC_THRESHOLD,
            'use_technical': cls.DEFAULT_USE_TECHNICAL,
            'use_aesthetic': cls.DEFAULT_USE_AESTHETIC
        }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize NIMA tagger with both technical and aesthetic models."""
        # We'll initialize both models but use config to determine which to use
        super().__init__(model_id="nima_technical", config=config)
        
        # Get the aesthetic model too
        from ..model.model_registry import ModelRegistry
        aesthetic_info = ModelRegistry.get_model("nima_aesthetic")
        self.aesthetic_model = aesthetic_info["model"]
        self.aesthetic_processor = aesthetic_info["processor"]

    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        """Generate quality scores for a PIL Image."""
        try:
            results: Dict[str, List[TagResult]] = {}
            
            # Technical quality assessment
            if self.config['use_technical']:
                logger.info("Running technical quality assessment")
                tech_inputs = self.processor(image)
                tech_scores = self.model.predict(tech_inputs["pixel_values"], verbose=0)[0]
                tech_mean = float(np.sum(np.array([i+1 for i in range(10)]) * tech_scores))
                
                logger.info(f"Technical quality score distribution: {[f'{s:.3f}' for s in tech_scores]}")
                logger.info(f"Technical mean score: {tech_mean:.3f}")
                
                # Normalize confidence to 0-1 range
                tech_confidence = (tech_mean - 1.0) / 9.0  # Convert 1-10 range to 0-1
                tech_quality = "acceptable" if tech_mean >= self.config['technical_threshold'] else "poor"
                
                results["technical_quality"] = [
                    TagResult(
                        label=tech_quality,
                        confidence=tech_confidence,  # Now normalized to 0-1
                        category="technical_quality"
                    )
                ]
                logger.info(f"Technical quality assessment: {tech_quality} (normalized confidence: {tech_confidence:.3f})")
                
                # Add detailed technical scores (these are already 0-1 as they're probabilities)
                results["technical_scores"] = [
                    TagResult(
                        label=f"score_{i+1}",
                        confidence=float(score),
                        category="technical_distribution"
                    ) for i, score in enumerate(tech_scores)
                ]
            
            # Aesthetic quality assessment
            if self.config['use_aesthetic']:
                logger.info("Running aesthetic quality assessment")
                aes_inputs = self.aesthetic_processor(image)
                aes_scores = self.aesthetic_model.predict(aes_inputs["pixel_values"], verbose=0)[0]
                aes_mean = float(np.sum(np.array([i+1 for i in range(10)]) * aes_scores))
                
                logger.info(f"Aesthetic score distribution: {[f'{s:.3f}' for s in aes_scores]}")
                logger.info(f"Aesthetic mean score: {aes_mean:.3f}")
                
                # Normalize confidence to 0-1 range
                aes_confidence = (aes_mean - 1.0) / 9.0  # Convert 1-10 range to 0-1
                aes_quality = "aesthetic" if aes_mean >= self.config['aesthetic_threshold'] else "unaesthetic"
                
                results["aesthetic_quality"] = [
                    TagResult(
                        label=aes_quality,
                        confidence=aes_confidence,  # Now normalized to 0-1
                        category="aesthetic_quality"
                    )
                ]
                logger.info(f"Aesthetic quality assessment: {aes_quality} (normalized confidence: {aes_confidence:.3f})")
                
                # Add detailed aesthetic scores (these are already 0-1 as they're probabilities)
                results["aesthetic_scores"] = [
                    TagResult(
                        label=f"score_{i+1}",
                        confidence=float(score),
                        category="aesthetic_distribution"
                    ) for i, score in enumerate(aes_scores)
                ]
            
            # Overall quality assessment combining both metrics if both are used
            if self.config['use_technical'] and self.config['use_aesthetic']:
                is_good_technical = tech_mean >= self.config['technical_threshold']
                is_good_aesthetic = aes_mean >= self.config['aesthetic_threshold']
                
                overall_quality = "high_quality" if (is_good_technical and is_good_aesthetic) else "low_quality"
                # Average of both normalized confidences
                overall_confidence = (tech_confidence + aes_confidence) / 2.0
                
                results["overall_quality"] = [
                    TagResult(
                        label=overall_quality,
                        confidence=overall_confidence,
                        category="overall_quality"
                    )
                ]
                logger.info(f"Overall quality assessment: {overall_quality} (confidence: {overall_confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"NIMA assessment failed: {e}")
            raise RuntimeError(f"NIMA assessment failed: {e}")