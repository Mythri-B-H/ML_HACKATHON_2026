"""
PathSnap - ML Feature Extractor
Extracts 9 geometric and kinematic features for candidate arc and GPS point pairs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from backend.src.utils.geo import angular_difference

FEATURE_NAMES = [
    "perp_dist",
    "heading_diff",
    "proj_ratio",
    "dist_to_u",
    "dist_to_v",
    "arc_length",
    "spatial_rank",
    "speed_mps",
    "speed_consistency"
]

def extract_candidate_features(candidate: Dict[str, Any],
                               gps_point: Dict[str, Any],
                               rank: int) -> List[float]:
    """
    Extracts the 9-dimensional feature vector for a single (candidate, GPS point) pair.
    """
    perp_dist = float(candidate["perp_dist"])
    arc_bearing = float(candidate["arc_bearing"])
    gps_bearing = float(gps_point.get("bearing", 0.0))
    heading_diff = angular_difference(gps_bearing, arc_bearing)
    
    proj_ratio = float(candidate["proj_ratio"])
    dist_to_u = float(candidate["dist_to_u"])
    dist_to_v = float(candidate["dist_to_v"])
    arc_length = float(candidate["arc_length"])
    
    speed_mps = float(gps_point.get("speed_mps", 0.0))
    # Ratio between instantaneous speed and arc length scale
    speed_consistency = speed_mps / (arc_length + 10.0) if arc_length > 0 else 0.0
    
    return [
        perp_dist,
        heading_diff,
        proj_ratio,
        dist_to_u,
        dist_to_v,
        arc_length,
        float(rank),
        speed_mps,
        speed_consistency
    ]

def build_training_dataset(trajectory_bundles: List[Dict[str, Any]],
                           search_radius_m: float = 60.0,
                           max_samples_per_traj: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs feature matrix X and binary target vector y from trajectory bundles.
    Ground truth label y=1 if candidate arc_id is in the trajectory's ground truth route.
    """
    from backend.src.network.graph import RoadNetwork
    from backend.src.network.spatial_index import SpatialGridIndex
    from backend.src.data.cleaner import clean_trajectory
    
    X_rows = []
    y_labels = []
    
    for bundle in trajectory_bundles:
        route_set = set(bundle["route"])
        if not route_set:
            continue
            
        network = RoadNetwork(bundle["nodes"], bundle["arcs"])
        spatial_index = SpatialGridIndex(network, cell_size_meters=100.0)
        cleaned_track, _ = clean_trajectory(bundle["track"])
        
        # Subsample points if trajectory is very long for balanced training time
        step = max(1, len(cleaned_track) // max_samples_per_traj)
        sampled_track = cleaned_track[::step]
        
        for pt in sampled_track:
            candidates = spatial_index.query_candidates(
                pt["smooth_lon"], pt["smooth_lat"], search_radius_m=search_radius_m, top_k=10
            )
            for rank, cand in enumerate(candidates):
                feats = extract_candidate_features(cand, pt, rank)
                label = 1 if cand["arc_id"] in route_set else 0
                X_rows.append(feats)
                y_labels.append(label)
                
    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,))
        
    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int32)
