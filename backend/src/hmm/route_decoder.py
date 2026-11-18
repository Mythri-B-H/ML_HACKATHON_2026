"""
PathSnap - Route Decoder & Topological Path Completer
Converts Viterbi candidate sequences into continuous road geometries,
bridges topological gaps via shortest path routing, and evaluates metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
from backend.src.network.graph import RoadNetwork
from backend.src.utils.geo import haversine_distance

class RouteDecoder:
    def __init__(self, road_network: RoadNetwork):
        self.network = road_network
        
    def decode_and_bridge_route(self, matched_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compresses repeated consecutive arcs, bridges gaps via shortest path Dijkstra,
        and generates continuous geographic coordinates.
        """
        if not matched_candidates:
            return {
                "arc_sequence": [],
                "continuous_arcs": [],
                "coordinates": [],
                "total_distance_km": 0.0,
                "num_gaps_bridged": 0
            }
            
        # 1. Extract raw matched arc IDs
        raw_arc_ids = [c["arc_id"] for c in matched_candidates]
        
        # 2. Collapse consecutive duplicates
        compressed_arcs = []
        for aid in raw_arc_ids:
            if not compressed_arcs or compressed_arcs[-1] != aid:
                compressed_arcs.append(aid)
                
        # 3. Bridge topological gaps between consecutive arcs
        continuous_arcs = []
        gaps_bridged = 0
        
        for i in range(len(compressed_arcs)):
            curr_arc = compressed_arcs[i]
            if not continuous_arcs:
                continuous_arcs.append(curr_arc)
                continue
                
            prev_arc = continuous_arcs[-1]
            u_prev, v_prev, _, _ = self.network.get_arc(prev_arc)
            u_curr, v_curr, _, _ = self.network.get_arc(curr_arc)
            
            if v_prev == u_curr:
                # Directly connected
                continuous_arcs.append(curr_arc)
            else:
                # Discontinuity detected: attempt shortest path bridge
                bridge_path = self.network.find_path_arcs(v_prev, u_curr, max_cutoff=2500.0)
                if bridge_path:
                    continuous_arcs.extend(bridge_path)
                    gaps_bridged += 1
                continuous_arcs.append(curr_arc)
                
        # 4. Generate continuous coordinates [ [lon, lat], ... ]
        coords = []
        total_dist_m = 0.0
        
        for aid in continuous_arcs:
            u, v, length_m, _ = self.network.get_arc(aid)
            u_lon, u_lat = self.network.get_node(u)
            v_lon, v_lat = self.network.get_node(v)
            
            if not coords:
                coords.append([u_lon, u_lat])
            coords.append([v_lon, v_lat])
            total_dist_m += length_m
            
        return {
            "arc_sequence": compressed_arcs,
            "continuous_arcs": continuous_arcs,
            "coordinates": coords,
            "total_distance_km": round(total_dist_m / 1000.0, 3),
            "num_gaps_bridged": gaps_bridged
        }

    def evaluate_route(self, predicted_arcs: List[int],
                       ground_truth_arcs: List[int]) -> Dict[str, float]:
        """
        Evaluates predicted arc sequence against ground truth .route sequence.
        Computes precision, recall, F1, and sequence matching accuracy.
        """
        if not ground_truth_arcs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "jaccard": 0.0}
            
        pred_set = set(predicted_arcs)
        gt_set = set(ground_truth_arcs)
        
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = tp / len(pred_set.union(gt_set)) if pred_set.union(gt_set) else 0.0
        
        # Sequence matching (Longest Common Subsequence or exact position matching)
        # Ratio of matching elements in order
        min_len = min(len(predicted_arcs), len(ground_truth_arcs))
        matches = sum(1 for i in range(min_len) if predicted_arcs[i] == ground_truth_arcs[i])
        seq_acc = matches / max(len(predicted_arcs), len(ground_truth_arcs)) if max(len(predicted_arcs), len(ground_truth_arcs)) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "jaccard_similarity": round(jaccard, 4),
            "sequence_accuracy": round(seq_acc, 4)
        }
