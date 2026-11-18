"""
PathSnap - Road Network Graph Representation
Directed road network graph with Haversine segment lengths,
adjacency indexing, and fast Dijkstra shortest path routing.
"""

import math
import heapq
from typing import List, Tuple, Dict, Optional, Set
from backend.src.utils.geo import haversine_distance, calculate_bearing

class RoadNetwork:
    """
    Directed Road Network Graph for map-matching.
    Vertices: Nodes (lon, lat)
    Edges: Directed Arcs (u -> v) with arc_id, length_meters, bearing_degrees
    """
    def __init__(self, nodes: List[Tuple[float, float]], arcs: List[Tuple[int, int]]):
        self.nodes = nodes  # index = node_id, value = (lon, lat)
        self.arcs = arcs    # index = arc_id, value = (u, v)
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)
        
        # Precompute arc attributes: (u, v, length_m, bearing_deg)
        self.arc_attributes = []
        # Adjacency list: u -> list of (v, arc_id, length_m)
        self.adj = [[] for _ in range(self.num_nodes)]
        # Reverse adjacency: v -> list of (u, arc_id, length_m)
        self.rev_adj = [[] for _ in range(self.num_nodes)]
        
        self._build_graph()
        self._dist_cache = {}  # Cache for repeated shortest-path queries (u, v)
        
    def _build_graph(self):
        for arc_id, (u, v) in enumerate(self.arcs):
            if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
                u_lon, u_lat = self.nodes[u]
                v_lon, v_lat = self.nodes[v]
                length = haversine_distance(u_lon, u_lat, v_lon, v_lat)
                bearing = calculate_bearing(u_lon, u_lat, v_lon, v_lat)
                self.arc_attributes.append((u, v, length, bearing))
                self.adj[u].append((v, arc_id, length))
                self.rev_adj[v].append((u, arc_id, length))
            else:
                self.arc_attributes.append((u, v, 0.0, 0.0))
                
    def get_node(self, node_id: int) -> Tuple[float, float]:
        return self.nodes[node_id]
        
    def get_arc(self, arc_id: int) -> Tuple[int, int, float, float]:
        """
        Returns (u, v, length_meters, bearing_degrees)
        """
        return self.arc_attributes[arc_id]

    def shortest_path_distance(self, start_node: int, end_node: int, max_cutoff_dist: float = 3000.0) -> float:
        """
        Computes shortest path distance between start_node and end_node using Dijkstra.
        Returns distance in meters (or max_cutoff_dist if unreachable or exceeding cutoff).
        """
        if start_node == end_node:
            return 0.0
            
        cache_key = (start_node, end_node)
        if cache_key in self._dist_cache:
            return self._dist_cache[cache_key]
            
        # Dijkstra search
        dist = {start_node: 0.0}
        pq = [(0.0, start_node)]
        
        found_dist = max_cutoff_dist
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == end_node:
                found_dist = d
                break
            if d > max_cutoff_dist:
                break
                
            for v, arc_id, length in self.adj[u]:
                new_d = d + length
                if new_d < dist.get(v, float('inf')) and new_d <= max_cutoff_dist:
                    dist[v] = new_d
                    heapq.heappush(pq, (new_d, v))
                    
        self._dist_cache[cache_key] = found_dist
        return found_dist

    def arc_transition_distance(self, arc1_id: int, arc2_id: int, max_cutoff: float = 3000.0) -> float:
        """
        Computes topological travel distance between arc1 and arc2:
        - If arc1 == arc2: distance = 0.0
        - If target(arc1) == source(arc2): consecutive arcs, distance = 0.0
        - Otherwise: shortest path from target(arc1) to source(arc2).
        """
        if arc1_id == arc2_id:
            return 0.0
            
        u1, v1, len1, _ = self.arc_attributes[arc1_id]
        u2, v2, len2, _ = self.arc_attributes[arc2_id]
        
        if v1 == u2:
            return 0.0  # Directly connected head-to-tail
            
        return self.shortest_path_distance(v1, u2, max_cutoff_dist=max_cutoff)
        
    def find_path_arcs(self, start_node: int, end_node: int, max_cutoff: float = 3000.0) -> List[int]:
        """
        Returns list of arc_ids on the shortest path from start_node to end_node.
        """
        if start_node == end_node:
            return []
            
        dist = {start_node: 0.0}
        parent = {}  # node -> (prev_node, arc_id)
        pq = [(0.0, start_node)]
        
        reached = False
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == end_node:
                reached = True
                break
            if d > max_cutoff:
                break
                
            for v, arc_id, length in self.adj[u]:
                new_d = d + length
                if new_d < dist.get(v, float('inf')) and new_d <= max_cutoff:
                    dist[v] = new_d
                    parent[v] = (u, arc_id)
                    heapq.heappush(pq, (new_d, v))
                    
        if not reached:
            return []
            
        path_arcs = []
        curr = end_node
        while curr != start_node:
            prev_node, aid = parent[curr]
            path_arcs.append(aid)
            curr = prev_node
        path_arcs.reverse()
        return path_arcs
