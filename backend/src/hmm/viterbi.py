"""
PathSnap - Viterbi Sequence Decoder
Finds the Maximum A Posteriori (MAP) sequence of road arcs across GPS observations.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from backend.src.network.graph import RoadNetwork
from backend.src.hmm.emission import EmissionModel
from backend.src.hmm.transition import TransitionModel

class ViterbiDecoder:
    def __init__(self, emission_model: Optional[EmissionModel] = None,
                 transition_model: Optional[TransitionModel] = None):
        self.emission_model = emission_model or EmissionModel()
        self.transition_model = transition_model or TransitionModel()
        
    def decode(self, road_network: RoadNetwork,
               cleaned_gps_points: List[Dict[str, Any]],
               candidates_per_step: List[List[Dict[str, Any]]],
               ml_scores_per_step: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Runs Viterbi dynamic programming in log-space.
        
        Returns:
            optimal_candidates: List of chosen candidate dicts for each time step.
        """
        T = len(cleaned_gps_points)
        if T == 0:
            return []
            
        # Viterbi tables
        # V[t][j] = max log-probability ending at candidate j at step t
        # backpointer[t][j] = index of best candidate at step t-1
        V = []
        backpointer = []
        
        # t = 0 (Initialization)
        cands_0 = candidates_per_step[0]
        if not cands_0:
            return []
            
        v_0 = []
        for j, cand in enumerate(cands_0):
            ml_score = ml_scores_per_step[0][j] if ml_scores_per_step else None
            p_emit = self.emission_model.compute_emission_prob(cand, ml_score)
            v_0.append(math.log(p_emit))
            
        V.append(v_0)
        backpointer.append([-1] * len(cands_0))
        
        # t = 1 to T-1 (Induction)
        for t in range(1, T):
            curr_cands = candidates_per_step[t]
            prev_cands = candidates_per_step[t - 1]
            curr_gps = cleaned_gps_points[t]
            prev_gps = cleaned_gps_points[t - 1]
            
            v_t = []
            bp_t = []
            
            for j, c_curr in enumerate(curr_cands):
                ml_score = ml_scores_per_step[t][j] if ml_scores_per_step else None
                log_p_emit = math.log(self.emission_model.compute_emission_prob(c_curr, ml_score))
                
                best_log_prob = float("-inf")
                best_i = 0
                
                for i, c_prev in enumerate(prev_cands):
                    p_trans = self.transition_model.compute_transition_prob(
                        road_network, c_prev, c_curr, prev_gps, curr_gps
                    )
                    log_p_trans = math.log(p_trans)
                    log_prob = V[t - 1][i] + log_p_trans + log_p_emit
                    
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_i = i
                        
                v_t.append(best_log_prob)
                bp_t.append(best_i)
                
            V.append(v_t)
            backpointer.append(bp_t)
            
        # Termination: Find best ending state at T-1
        best_last_idx = int(np.argmax(V[-1]))
        
        # Backtracking
        optimal_candidates = [None] * T
        curr_best = best_last_idx
        for t in range(T - 1, -1, -1):
            optimal_candidates[t] = candidates_per_step[t][curr_best]
            curr_best = backpointer[t][curr_best]
            
        return optimal_candidates
