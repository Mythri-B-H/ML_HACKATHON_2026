"""
Unit Tests for Data Parsers
"""

import os
import pytest
from backend.src.data.parser import parse_track, parse_nodes, parse_arcs, parse_route

DATA_SAMPLE_DIR = r"c:\Users\bhmyt\PathSnap\data\samples\00000001"

def test_parse_track():
    track_file = os.path.join(DATA_SAMPLE_DIR, "00000001.track")
    points = parse_track(track_file)
    assert len(points) == 1315
    first = points[0]
    assert "lon" in first and "lat" in first and "time" in first
    assert abs(first["lon"] - 8.595203) < 1e-5
    assert abs(first["lat"] - 49.844053) < 1e-5
    assert first["time"] == 1.0

def test_parse_nodes():
    nodes_file = os.path.join(DATA_SAMPLE_DIR, "00000001.nodes")
    nodes = parse_nodes(nodes_file)
    assert len(nodes) == 88809
    assert isinstance(nodes[0], tuple)
    assert len(nodes[0]) == 2

def test_parse_arcs():
    arcs_file = os.path.join(DATA_SAMPLE_DIR, "00000001.arcs")
    arcs = parse_arcs(arcs_file)
    assert len(arcs) == 165790
    assert isinstance(arcs[0], tuple)
    assert arcs[0] == (0, 281)

def test_parse_route():
    route_file = os.path.join(DATA_SAMPLE_DIR, "00000001.route")
    route = parse_route(route_file)
    assert len(route) == 525
    assert route[0] == 2641
    assert route[1] == 64973
