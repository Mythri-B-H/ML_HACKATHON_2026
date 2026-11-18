"""
PathSnap - End-to-End Route Reconstruction Engine
Orchestrates candidate generation, ML scoring, HMM/Viterbi decoding,
continuous route interpolation, and ML vs. HMM comparison.
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np

from backend.src.network.graph import RoadNetwork
from backend.src.network.spatial_index import SpatialGridIndex
from backend.src.data.cleaner import clean_trajectory
from backend.src.ml.features import extract_candidate_features
from backend.src.ml.models import BaseRoadModel
from backend.src.hmm.emission import EmissionModel
from backend.src.hmm.transition import TransitionModel
from backend.src.hmm.viterbi import ViterbiDecoder
from backend.src.hmm.route_decoder import RouteDecoder

class RouteReconstructionPipeline:
    def __init__(self, road_network: RoadNetwork,
                 spatial_index: Optional[SpatialGridIndex] = None,
                 ml_model: Optional[BaseRoadModel] = None,
                 search_radius_m: float = 60.0,
                 beta: float = 25.0,
                 sigma_z: float = 12.0):
        self.network = road_network
        self.spatial_index = spatial_index or SpatialGridIndex(road_network)
        self.ml_model = ml_model
        self.search_radius_m = search_radius_m
        
        self.emission_model = EmissionModel(sigma_z=sigma_z, ml_weight=0.6 if ml_model else 0.0)
        self.transition_model = TransitionModel(beta=beta)
        self.viterbi = ViterbiDecoder(self.emission_model, self.transition_model)
        self.route_decoder = RouteDecoder(road_network)
        
    def reconstruct(self, raw_track_points: List[Dict[str, float]],
                    ground_truth_route: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Runs the full route reconstruction pipeline on raw GPS track points.
        """
        t0 = time.time()
        
        # 1. Cleaning & Kinematics
        cleaned_points, audit = clean_trajectory(raw_track_points)
        if not cleaned_points:
            return {"status": "error", "message": "No valid GPS points"}
            
        # 2. Candidate Generation & Feature Extraction
        candidates_per_step = []
        ml_scores_per_step = []
        ml_only_preds = []
        
        for pt in cleaned_points:
            cands = self.spatial_index.query_candidates(
                pt["smooth_lon"], pt["smooth_lat"],
                search_radius_m=self.search_radius_m, top_k=10
            )
            candidates_per_step.append(cands)
            
            # ML Scoring
            if self.ml_model and cands:
                feats = [extract_candidate_features(c, pt, rank) for rank, c in enumerate(cands)]
                feats_arr = np.array(feats, dtype=np.float32)
                probs = self.ml_model.predict_proba(feats_arr)
                match_probs = probs[:, 1].tolist()
                ml_scores_per_step.append(match_probs)
                
                # ML-only top candidate prediction
                best_cand_idx = int(np.argmax(match_probs))
                ml_only_preds.append(cands[best_cand_idx]["arc_id"])
            elif cands:
                # Nearest candidate by distance
                ml_scores_per_step.append([1.0 / (c["perp_dist"] + 1.0) for c in cands])
                ml_only_preds.append(cands[0]["arc_id"])
            else:
                ml_scores_per_step.append([])
                
        # 3. HMM + Viterbi Decoding
        viterbi_matched = self.viterbi.decode(
            self.network, cleaned_points, candidates_per_step, ml_scores_per_step
        )
        
        # 4. Continuous Route Bridging
        reconstructed = self.route_decoder.decode_and_bridge_route(viterbi_matched)
        
        total_time_ms = round((time.time() - t0) * 1000.0, 2)
        
        # 5. ML-only vs. HMM Comparison & Evaluation
        evaluation = None
        ml_only_eval = None
        if ground_truth_route:
            evaluation = self.route_decoder.evaluate_route(
                reconstructed["continuous_arcs"], ground_truth_route
            )
            # Compress ML-only arc sequence for fair comparison
            ml_compressed = []
            for aid in ml_only_preds:
                if not ml_compressed or ml_compressed[-1] != aid:
                    ml_compressed.append(aid)
            ml_only_eval = self.route_decoder.evaluate_route(
                ml_compressed, ground_truth_route
            )
            
        return {
            "status": "success",
            "processing_time_ms": total_time_ms,
            "num_gps_raw": len(raw_track_points),
            "num_gps_cleaned": len(cleaned_points),
            "cleaning_audit": audit,
            "reconstructed_distance_km": reconstructed["total_distance_km"],
            "continuous_arcs": reconstructed["continuous_arcs"],
            "arc_sequence": reconstructed["arc_sequence"],
            "coordinates": reconstructed["coordinates"],
            "num_gaps_bridged": reconstructed["num_gaps_bridged"],
            "evaluation": evaluation,
            "ml_only_evaluation": ml_only_eval,
            "used_ml_model": self.ml_model.name if self.ml_model else "None (Pure HMM)"
        }
