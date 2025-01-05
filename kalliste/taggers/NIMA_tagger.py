"""Image quality assessment using NIMA (Neural Image Assessment)."""
from typing import Dict, List, Optional
from PIL import Image
import logging
import numpy as np

from .AbstractTagger import AbstractTagger
from ..types import TagResult

logger = logging.getLogger(__name__)

class NIMATagger(AbstractTagger):
    # Updated thresholds now in 0-1 space
    DEFAULT_LOW_THRESHOLD = 0.45
    DEFAULT_HIGH_THRESHOLD = 0.55
    
    def _normalize_score(self, mean_score: float) -> float:
        """Convert 1-10 scale score to 0-1 scale."""
        return (mean_score - 1.0) / 9.0
        
    async def tag_pillow_image(self, image: Image.Image) -> Dict[str, List[TagResult]]:
        results = {}
        
        # Technical assessment
        if self.config['use_technical']:
            tech_scores = self.model.predict(...)
            tech_mean = float(np.sum(np.array([i+1 for i in range(10)]) * tech_scores))
            tech_normalized = self._normalize_score(tech_mean)
            
            # Determine technical assessment based on thresholds
            if tech_normalized < self.config.get('technical_low_threshold', self.DEFAULT_LOW_THRESHOLD):
                tech_assessment = "poor_quality"
            elif tech_normalized > self.config.get('technical_high_threshold', self.DEFAULT_HIGH_THRESHOLD):
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
            
        # Aesthetic assessment (similar structure)
        if self.config['use_aesthetic']:
            aes_scores = self.aesthetic_model.predict(...)
            aes_mean = float(np.sum(np.array([i+1 for i in range(10)]) * aes_scores))
            aes_normalized = self._normalize_score(aes_mean)
            
            if aes_normalized < self.config.get('aesthetic_low_threshold', self.DEFAULT_LOW_THRESHOLD):
                aes_assessment = "tasteless"
            elif aes_normalized > self.config.get('aesthetic_high_threshold', self.DEFAULT_HIGH_THRESHOLD):
                aes_assessment = "aesthetic"
            else:
                aes_assessment = "middling"

            # Store results similar to technical...

        # Calculate combined scores
        if self.config['use_technical'] and self.config['use_aesthetic']:
            # Simple average
            calc_average = (tech_normalized + aes_normalized) / 2.0
            
            # Combined distribution (normalized probabilities)
            combined_dist = [(t + a)/2 for t, a in zip(tech_scores, aes_scores)]
            
            # Overall assessment
            is_tech_acceptable = tech_assessment in ["medium_quality", "high_quality"]
            is_aes_acceptable = aes_assessment in ["middling", "aesthetic"]
            overall = "acceptable" if (is_tech_acceptable and is_aes_acceptable) else "unacceptable"

            results["overall_calculations"] = [
                TagResult(label="average", confidence=calc_average, category="calc_average"),
                TagResult(label="distribution", confidence=combined_dist, category="calc_distribution")
            ]
            
            results["overall_assessment"] = [
                TagResult(label=overall, confidence=calc_average, category="overall")
            ]