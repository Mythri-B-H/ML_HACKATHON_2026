"""
PathSnap - HMM Emission Probability Estimator
Combines perpendicular distance Gaussian prior with calibrated ML candidate match probabilities.
"""

import math
from typing import Dict, Any, Optional
import numpy as np

class EmissionModel:
    def __init__(self, sigma_z: float = 12.0, ml_weight: float = 0.6):
        """
        sigma_z: Standard deviation of GPS perpendicular measurement noise in meters.
        ml_weight: Weight given to ML model probability [0.0 = pure Gaussian, 1.0 = pure ML].
        """
        self.sigma_z = sigma_z
        self.ml_weight = ml_weight
        self.two_sigma_sq = 2.0 * (sigma_z ** 2)
        self.norm_const = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_z)
        
    def compute_emission_prob(self, candidate: Dict[str, Any],
                              ml_score: Optional[float] = None) -> float:
        """
        Calculates P(z_t | c_ti).
        """
        perp_dist = candidate["perp_dist"]
        p_gaussian = self.norm_const * math.exp(-(perp_dist ** 2) / self.two_sigma_sq)
        
        if ml_score is None:
            return max(1e-12, p_gaussian)
            
        p_ml = max(1e-6, min(1.0 - 1e-6, ml_score))
        # Blend Gaussian geometric prior with ML prediction
        prob = ((1.0 - self.ml_weight) * p_gaussian) + (self.ml_weight * p_ml * self.norm_const)
        return max(1e-12, prob)
