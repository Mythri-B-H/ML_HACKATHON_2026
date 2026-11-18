"""
PathSnap - HMM Transition Probability Estimator
Estimates P(c_t,j | c_{t-1},i) based on difference between
road network shortest path distance and great-circle GPS displacement.
"""

import math
from typing import Dict, Any
from backend.src.network.graph import RoadNetwork
from backend.src.utils.geo import haversine_distance

class TransitionModel:
    def __init__(self, beta: float = 25.0, max_cutoff_dist: float = 2000.0):
        """
        beta: Exponential distribution parameter scale (meters).
        max_cutoff_dist: Maximum graph search cutoff.
        """
        self.beta = beta
        self.max_cutoff = max_cutoff_dist
        
    def compute_transition_prob(self, road_network: RoadNetwork,
                                prev_candidate: Dict[str, Any],
                                curr_candidate: Dict[str, Any],
                                prev_gps: Dict[str, Any],
                                curr_gps: Dict[str, Any]) -> float:
        """
        Calculates P(c_t,j | c_{t-1},i).
        """
        arc_prev = prev_candidate["arc_id"]
        arc_curr = curr_candidate["arc_id"]
        
        # Great-circle distance between consecutive GPS points
        d_gps = haversine_distance(
            prev_gps["smooth_lon"], prev_gps["smooth_lat"],
            curr_gps["smooth_lon"], curr_gps["smooth_lat"]
        )
        
        # Topological network distance between candidate arcs
        d_network = road_network.arc_transition_distance(arc_prev, arc_curr, max_cutoff=self.max_cutoff)
        
        # Discrepancy delta_d
        delta_d = abs(d_network - d_gps)
        
        # Exponential distribution probability density
        prob = (1.0 / self.beta) * math.exp(-delta_d / self.beta)
        return max(1e-12, prob)
