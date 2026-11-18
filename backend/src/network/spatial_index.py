"""
PathSnap - Spatial Index for Fast Candidate Arc Generation
Uses uniform spatial grid partitioning with bounding boxes
for sub-millisecond candidate road queries.
"""

import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from backend.src.network.graph import RoadNetwork
from backend.src.utils.geo import (
    EARTH_RADIUS_METERS,
    project_point_to_segment,
    haversine_distance
)

class SpatialGridIndex:
    """
    Uniform spatial grid index for directed road segments.
    Provides fast spatial pruning before exact orthogonal projection calculations.
    """
    def __init__(self, road_network: RoadNetwork, cell_size_meters: float = 100.0):
        self.network = road_network
        self.cell_size_m = cell_size_meters
        self.grid = defaultdict(list)  # (cell_x, cell_y) -> list of arc_ids
        
        # Determine origin for local grid projection
        if road_network.nodes:
            self.origin_lon = road_network.nodes[0][0]
            self.origin_lat = road_network.nodes[0][1]
        else:
            self.origin_lon, self.origin_lat = 0.0, 0.0
            
        self.cos_origin_lat = math.cos(math.radians(self.origin_lat))
        self._build_index()
        
    def _to_grid_coords(self, lon: float, lat: float) -> Tuple[int, int]:
        x = math.radians(lon - self.origin_lon) * EARTH_RADIUS_METERS * self.cos_origin_lat
        y = math.radians(lat - self.origin_lat) * EARTH_RADIUS_METERS
        gx = int(math.floor(x / self.cell_size_m))
        gy = int(math.floor(y / self.cell_size_m))
        return gx, gy

    def _build_index(self):
        for arc_id, (u, v) in enumerate(self.network.arcs):
            if u >= self.network.num_nodes or v >= self.network.num_nodes:
                continue
            u_lon, u_lat = self.network.nodes[u]
            v_lon, v_lat = self.network.nodes[v]
            
            gx1, gy1 = self._to_grid_coords(u_lon, u_lat)
            gx2, gy2 = self._to_grid_coords(v_lon, v_lat)
            
            min_gx, max_gx = min(gx1, gx2), max(gx1, gx2)
            min_gy, max_gy = min(gy1, gy2), max(gy1, gy2)
            
            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    self.grid[(gx, gy)].append(arc_id)

    def query_candidates(self, gps_lon: float, gps_lat: float,
                         search_radius_m: float = 60.0,
                         top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Queries candidate road segments near (gps_lon, gps_lat).
        Returns top_k candidate arc dicts sorted ascending by perpendicular distance.
        """
        center_gx, center_gy = self._to_grid_coords(gps_lon, gps_lat)
        cell_radius = int(math.ceil(search_radius_m / self.cell_size_m))
        
        candidate_arc_ids = set()
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (center_gx + dx, center_gy + dy)
                if cell_key in self.grid:
                    for aid in self.grid[cell_key]:
                        candidate_arc_ids.add(aid)
                        
        candidates = []
        for aid in candidate_arc_ids:
            u, v, length_m, bearing_deg = self.network.get_arc(aid)
            u_lon, u_lat = self.network.nodes[u]
            v_lon, v_lat = self.network.nodes[v]
            
            perp_dist, proj_ratio, proj_lon, proj_lat, dist_u, dist_v = project_point_to_segment(
                gps_lon, gps_lat, u_lon, u_lat, v_lon, v_lat
            )
            
            if perp_dist <= search_radius_m:
                candidates.append({
                    "arc_id": aid,
                    "u": u,
                    "v": v,
                    "perp_dist": perp_dist,
                    "proj_ratio": proj_ratio,
                    "proj_lon": proj_lon,
                    "proj_lat": proj_lat,
                    "dist_to_u": dist_u,
                    "dist_to_v": dist_v,
                    "arc_length": length_m,
                    "arc_bearing": bearing_deg
                })
                
        # Sort ascending by perpendicular distance
        candidates.sort(key=lambda c: c["perp_dist"])
        
        # If no candidates found within search_radius_m, fallback to nearest arc within 2x radius
        if not candidates and search_radius_m < 200.0:
            return self.query_candidates(gps_lon, gps_lat, search_radius_m=search_radius_m * 2.0, top_k=top_k)
            
        return candidates[:top_k]
