"""
PathSnap - Geographic and Spatial Utilities
Accurate Haversine distances, spherical bearings, and orthogonal projections.
"""

import math
from typing import Tuple

EARTH_RADIUS_METERS = 6371000.0

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes Great-Circle (Haversine) distance between two points in meters.
    """
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c

def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes initial compass bearing from (lon1, lat1) to (lon2, lat2) in degrees [0, 360).
    """
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg

def angular_difference(bearing1: float, bearing2: float) -> float:
    """
    Computes absolute minimum angular difference between two bearings in degrees [0, 180].
    """
    diff = abs(bearing1 - bearing2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff

def project_point_to_segment(p_lon: float, p_lat: float,
                             a_lon: float, a_lat: float,
                             b_lon: float, b_lat: float) -> Tuple[float, float, float, float, float, float]:
    """
    Projects a GPS point onto a directed line segment (A -> B).
    Uses local equirectangular metric projection around p_lat.
    
    Returns:
        perp_dist: Distance from point to closest point on segment in meters.
        proj_ratio: Normalized scalar projection t clamped to [0, 1].
        proj_lon: Longitude of projected point.
        proj_lat: Latitude of projected point.
        dist_to_a: Distance from point to start node A in meters.
        dist_to_b: Distance from point to end node B in meters.
    """
    cos_lat = math.cos(math.radians(p_lat))
    x0 = math.radians(p_lon) * EARTH_RADIUS_METERS * cos_lat
    y0 = math.radians(p_lat) * EARTH_RADIUS_METERS
    x1 = math.radians(a_lon) * EARTH_RADIUS_METERS * cos_lat
    y1 = math.radians(a_lat) * EARTH_RADIUS_METERS
    x2 = math.radians(b_lon) * EARTH_RADIUS_METERS * cos_lat
    y2 = math.radians(b_lat) * EARTH_RADIUS_METERS
    
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    
    dist_to_a = math.hypot(x0 - x1, y0 - y1)
    dist_to_b = math.hypot(x0 - x2, y0 - y2)
    
    if seg_len_sq < 1e-6:
        # Segment is degenerate (start == end)
        return dist_to_a, 0.0, a_lon, a_lat, dist_to_a, dist_to_b
        
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    
    closest_x = x1 + t_clamped * dx
    closest_y = y1 + t_clamped * dy
    perp_dist = math.hypot(x0 - closest_x, y0 - closest_y)
    
    # Convert closest metric point back to lon, lat
    proj_lat = closest_y / EARTH_RADIUS_METERS * (180.0 / math.pi)
    proj_lon = (closest_x / (EARTH_RADIUS_METERS * cos_lat)) * (180.0 / math.pi) if abs(cos_lat) > 1e-7 else a_lon
    
    return perp_dist, t_clamped, proj_lon, proj_lat, dist_to_a, dist_to_b
