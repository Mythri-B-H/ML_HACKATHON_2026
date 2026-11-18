"""
Unit Tests for Spatial Candidate Indexing
"""

import pytest
from backend.src.network.graph import RoadNetwork
from backend.src.network.spatial_index import SpatialGridIndex

def test_spatial_grid_candidates():
    nodes = [
        (8.590, 49.840),
        (8.591, 49.840),
        (8.590, 49.850),  # far away
        (8.591, 49.850)
    ]
    arcs = [
        (0, 1),  # near
        (2, 3)   # far
    ]
    net = RoadNetwork(nodes, arcs)
    idx = SpatialGridIndex(net, cell_size_meters=100.0)
    
    # Query point very close to Arc 0
    cands = idx.query_candidates(8.5905, 49.8401, search_radius_m=50.0)
    assert len(cands) >= 1
    assert cands[0]["arc_id"] == 0
    assert cands[0]["perp_dist"] < 25.0
