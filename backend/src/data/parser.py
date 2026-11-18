"""
PathSnap - GPS, Road Network, and Route Parser
Parses .track, .nodes, .arcs, and .route files.
"""

import os
from typing import List, Tuple, Dict, Any, Optional

def parse_track(file_path: str) -> List[Dict[str, float]]:
    """
    Parses a .track file.
    Format: longitude \\t latitude \\t timestamp
    Returns: List of dicts with 'lon', 'lat', 'time', 'point_idx'
    """
    points = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Track file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                time_sec = float(parts[2])
                points.append({
                    "point_idx": idx,
                    "lon": lon,
                    "lat": lat,
                    "time": time_sec
                })
            except ValueError:
                continue
    return points

def parse_nodes(file_path: str) -> List[Tuple[float, float]]:
    """
    Parses a .nodes file.
    Format: longitude \\t latitude
    Returns: List of (lon, lat) tuples where list index = Node ID.
    """
    nodes = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Nodes file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                nodes.append((lon, lat))
            except ValueError:
                continue
    return nodes

def parse_arcs(file_path: str) -> List[Tuple[int, int]]:
    """
    Parses an .arcs file.
    Format: source_node_id \\t target_node_id
    Returns: List of (u, v) tuples where list index = Arc ID.
    """
    arcs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arcs file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                arcs.append((u, v))
            except ValueError:
                continue
    return arcs

def parse_route(file_path: str) -> List[int]:
    """
    Parses a .route file.
    Format: arc_id (one per line)
    Returns: List of traversed arc IDs in sequence.
    """
    route = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Route file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                arc_id = int(s)
                route.append(arc_id)
            except ValueError:
                continue
    return route

def load_trajectory_bundle(traj_dir: str, traj_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads all four files for a trajectory directory: .track, .nodes, .arcs, .route
    """
    if traj_id is None:
        traj_id = os.path.basename(traj_dir.rstrip("/\\"))
        
    track_path = os.path.join(traj_dir, f"{traj_id}.track")
    nodes_path = os.path.join(traj_dir, f"{traj_id}.nodes")
    arcs_path = os.path.join(traj_dir, f"{traj_id}.arcs")
    route_path = os.path.join(traj_dir, f"{traj_id}.route")
    
    track = parse_track(track_path)
    nodes = parse_nodes(nodes_path)
    arcs = parse_arcs(arcs_path)
    route = parse_route(route_path) if os.path.exists(route_path) else []
    
    return {
        "traj_id": traj_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }
