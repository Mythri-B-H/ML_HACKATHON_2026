"""
Unit Tests for Road Network Graph & Shortest Path
"""

import pytest
from backend.src.network.graph import RoadNetwork
from backend.src.utils.geo import haversine_distance

def test_road_network_dijkstra():
    # Simple linear graph: Node 0 -> Node 1 -> Node 2
    nodes = [
        (8.590, 49.840),  # Node 0
        (8.591, 49.840),  # Node 1 (~71m east)
        (8.592, 49.840),  # Node 2 (~71m east)
    ]
    arcs = [
        (0, 1),  # Arc 0
        (1, 2),  # Arc 1
    ]
    
    net = RoadNetwork(nodes, arcs)
    assert net.num_nodes == 3
    assert net.num_arcs == 2
    
    # Check shortest path from 0 to 2
    d = net.shortest_path_distance(0, 2)
    expected_d = haversine_distance(8.590, 49.840, 8.591, 49.840) + haversine_distance(8.591, 49.840, 8.592, 49.840)
    assert abs(d - expected_d) < 1.0
    
    # Check path arcs
    path = net.find_path_arcs(0, 2)
    assert path == [0, 1]
    
    # Check arc transition distance
    # Arc 0 -> Arc 1 (connected at node 1)
    trans_d = net.arc_transition_distance(0, 1)
    assert trans_d == 0.0

def test_disconnected_nodes():
    nodes = [(8.59, 49.84), (8.60, 49.84)]
    arcs = []  # No edges
    net = RoadNetwork(nodes, arcs)
    d = net.shortest_path_distance(0, 1, max_cutoff_dist=1000.0)
    assert d == 1000.0
