"""Image quality assessment using NIMA (Neural Image Assessment)."""
from typing import Dict, List, Optional
from PIL import Image
import logging
import numpy as np

from .AbstractTagger import AbstractTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class NIMATagger(AbstractTagger):
    # Class constants for thresholds
    DEFAULT_TECHNICAL_LOW_THRESHOLD = 0.52
    DEFAULT_TECHNICAL_HIGH_THRESHOLD = 0.56

    DEFAULT_AESTHETIC_LOW_THRESHOLD = 0.43
    DEFAULT_AESTHETIC_HIGH_THRESHOLD = 0.50
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize NIMA tagger with both technical and aesthetic models."""
        # Initialize technical model through parent class
        super().__init__(model_id="nima_technical", config=config)
        
        # Get the aesthetic model separately
        from ..model.model_registry import ModelRegistry
        aesthetic_info = ModelRegistry.get_model("nima_aesthetic")
        self.aesthetic_model = aesthetic_info["model"]
        self.aesthetic_processor = aesthetic_info["processor"]

    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default configuration for NIMA tagger."""
        return {
            'technical_low_threshold': cls.DEFAULT_TECHNICAL_LOW_THRESHOLD,
            'technical_high_threshold': cls.DEFAULT_TECHNICAL_HIGH_THRESHOLD,
            'aesthetic_low_threshold': cls.DEFAULT_AESTHETIC_LOW_THRESHOLD,
            'aesthetic_high_threshold': cls.DEFAULT_AESTHETIC_HIGH_THRESHOLD
        }

    def _normalize_score(self, mean_score: float) -> float:
        """Convert 1-10 scale score to 0-1 scale."""
        return (mean_score - 1.0) / 9.0
        
    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        """Generate quality scores for a PIL Image."""
        try:
            results: Dict[str, List[TagResult]] = {}
            
            # Technical assessment
            tech_inputs = self.processor(image)
            tech_scores = self.model.predict(tech_inputs["pixel_values"], verbose=0)[0]
            tech_mean = float(np.sum(np.array([i+1 for i in range(10)]) * tech_scores))
            tech_normalized = self._normalize_score(tech_mean)
            
            # Determine technical assessment based on thresholds
            if tech_normalized < self.config.get('technical_low_threshold', self.DEFAULT_TECHNICAL_LOW_THRESHOLD):
                tech_assessment = "poor_quality"
            elif tech_normalized > self.config.get('technical_high_threshold', self.DEFAULT_TECHNICAL_HIGH_THRESHOLD):
                tech_assessment = "high_quality"
            else:
                tech_assessment = "medium_quality"
                
            results["technical_score"] = [
                TagResult(
                    label=str(tech_normalized),
                    confidence=tech_normalized,
                    category="technical_score"
                )
            ]
            
            results["technical_assessment"] = [
                TagResult(
                    label=tech_assessment,
                    confidence=tech_normalized,
                    category="technical_assessment"
                )
            ]
        
            # Aesthetic assessment
            aes_inputs = self.aesthetic_processor(image)
            aes_scores = self.aesthetic_model.predict(aes_inputs["pixel_values"], verbose=0)[0]
            aes_mean = float(np.sum(np.array([i+1 for i in range(10)]) * aes_scores))
            aes_normalized = self._normalize_score(aes_mean)
            
            if aes_normalized < self.config.get('aesthetic_low_threshold', self.DEFAULT_AESTHETIC_LOW_THRESHOLD):
                aes_assessment = "tasteless"
            elif aes_normalized > self.config.get('aesthetic_high_threshold', self.DEFAULT_AESTHETIC_HIGH_THRESHOLD):
                aes_assessment = "aesthetic"
            else:
                aes_assessment = "middling"

            results["aesthetic_score"] = [
                TagResult(
                    label=str(aes_normalized),
                    confidence=aes_normalized,
                    category="aesthetic_score"
                )
            ]
            
            results["aesthetic_assessment"] = [
                TagResult(
                    label=aes_assessment,
                    confidence=aes_normalized,
                    category="aesthetic_assessment"
                )
            ]

            # Calculate combined assessments
            # Simple average
            calc_average = (tech_normalized + aes_normalized) / 2.0
            
            # Calculate score from combined distributions
            combined_dist = [(t + a)/2 for t, a in zip(tech_scores, aes_scores)]
            combined_mean = float(np.sum(np.array([i+1 for i in range(10)]) * combined_dist))
            combined_normalized = self._normalize_score(combined_mean)

            # Overall assessment
            is_tech_acceptable = tech_assessment in ["medium_quality", "high_quality"]
            is_aes_acceptable = aes_assessment in ["middling", "aesthetic"]
            overall = "acceptable" if (is_tech_acceptable and is_aes_acceptable) else "unacceptable"

            # Combined distribution (normalized probabilities) 
            combined_dist_str = ",".join(f"{x:.4f}" for x in combined_dist)

            results["overall_calculations"] = [
                TagResult(label="average", confidence=calc_average, category="calc_average"),
            ]
                            
            results["overall_assessment"] = [
                TagResult(label=overall, confidence=calc_average, category="overall")
            ]
        
            return results
            
        except Exception as e:
            logger.error(f"NIMA assessment failed: {e}")
            raise RuntimeError(f"NIMA assessment failed: {e}")
