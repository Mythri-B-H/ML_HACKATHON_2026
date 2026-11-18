"""
Unit Tests for HMM, Emission, Transition, and Route Reconstruction
"""

import pytest
from backend.src.network.graph import RoadNetwork
from backend.src.network.spatial_index import SpatialGridIndex
from backend.src.hmm.emission import EmissionModel
from backend.src.hmm.transition import TransitionModel
from backend.src.hmm.pipeline import RouteReconstructionPipeline

def test_emission_model():
    em = EmissionModel(sigma_z=10.0)
    cand_near = {"perp_dist": 2.0}
    cand_far = {"perp_dist": 30.0}
    p_near = em.compute_emission_prob(cand_near)
    p_far = em.compute_emission_prob(cand_far)
    assert p_near > p_far

def test_transition_model():
    nodes = [(8.59, 49.84), (8.591, 49.84), (8.592, 49.84)]
    arcs = [(0, 1), (1, 2)]
    net = RoadNetwork(nodes, arcs)
    tm = TransitionModel(beta=25.0)
    
    prev_c = {"arc_id": 0}
    curr_c = {"arc_id": 1}
    gps1 = {"smooth_lon": 8.5905, "smooth_lat": 49.84}
    gps2 = {"smooth_lon": 8.5915, "smooth_lat": 49.84}
    
    prob = tm.compute_transition_prob(net, prev_c, curr_c, gps1, gps2)
    assert prob > 0.0

def test_pipeline_reconstruct():
    nodes = [
        (8.590, 49.840),
        (8.591, 49.840),
        (8.592, 49.840)
    ]
    arcs = [(0, 1), (1, 2)]
    net = RoadNetwork(nodes, arcs)
    idx = SpatialGridIndex(net)
    pipeline = RouteReconstructionPipeline(net, idx)
    
    track = [
        {"lon": 8.5902, "lat": 49.84005, "time": 1.0},
        {"lon": 8.5908, "lat": 49.84004, "time": 2.0},
        {"lon": 8.5914, "lat": 49.84003, "time": 3.0},
        {"lon": 8.5918, "lat": 49.84001, "time": 4.0},
    ]
    gt_route = [0, 1]
    
    res = pipeline.reconstruct(track, gt_route)
    assert res["status"] == "success"
    assert res["continuous_arcs"] == [0, 1]
    assert len(res["coordinates"]) >= 3
    assert res["evaluation"]["recall"] == 1.0
