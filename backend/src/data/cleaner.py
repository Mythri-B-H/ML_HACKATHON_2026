"""
PathSnap - GPS Trajectory Cleaning, Kinematics, and Outlier Filtering
Detects duplicates, calculates speed/bearing, flags anomalies, and applies Kalman smoothing.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from backend.src.utils.geo import haversine_distance, calculate_bearing

class KalmanFilter2D:
    """
    2D Constant-Velocity Kinematic Kalman Filter for latitude/longitude smoothing.
    """
    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 25.0):
        self.q = process_noise
        self.r = measurement_noise
        self.x = None  # state vector: [lon, lat, vx, vy]
        self.p = np.eye(4) * 100.0  # covariance matrix
        
    def initialize(self, lon: float, lat: float):
        self.x = np.array([lon, lat, 0.0, 0.0], dtype=float)
        self.p = np.eye(4) * 10.0

    def step(self, lon: float, lat: float, dt: float) -> Tuple[float, float]:
        if self.x is None:
            self.initialize(lon, lat)
            return lon, lat
            
        dt = max(0.1, min(10.0, dt))
        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        # Process covariance Q
        q_pos = 0.5 * (dt**2) * self.q
        q_vel = dt * self.q
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        
        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.p @ F.T + Q
        
        # Measurement matrix H
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        # Measurement covariance R
        # Approx meters to degree conversion for noise
        R = np.eye(2) * (self.r / 111320.0)**2
        
        # Innovation
        z = np.array([lon, lat])
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update
        self.x = x_pred + (K @ y)
        self.p = (np.eye(4) - K @ H) @ P_pred
        
        return float(self.x[0]), float(self.x[1])

def clean_trajectory(raw_points: List[Dict[str, float]],
                     speed_limit_kmh: float = 150.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Cleans raw GPS track points, calculates kinematics, flags outliers,
    and produces Kalman-smoothed coordinates.
    
    Returns:
        cleaned_points: List of point dictionaries with enriched features.
        audit_report: Statistical metrics of the cleaning transformations.
    """
    if not raw_points:
        return [], {"status": "empty", "num_raw": 0, "num_cleaned": 0}
        
    cleaned = []
    audit = {
        "num_raw": len(raw_points),
        "num_cleaned": 0,
        "duplicate_coords_handled": 0,
        "speed_outliers_flagged": 0,
        "temporal_gaps_flagged": 0,
        "max_raw_speed_kmh": 0.0,
        "max_cleaned_speed_kmh": 0.0,
        "total_distance_meters": 0.0,
        "total_duration_seconds": 0.0
    }
    
    kf = KalmanFilter2D()
    prev_pt = None
    
    for i, pt in enumerate(raw_points):
        lon = float(pt["lon"])
        lat = float(pt["lat"])
        t = float(pt["time"])
        idx = pt.get("point_idx", i)
        
        is_duplicate = False
        is_speed_outlier = False
        is_temporal_gap = False
        dt = 1.0
        dist = 0.0
        speed_mps = 0.0
        speed_kmh = 0.0
        bearing = 0.0
        
        if prev_pt is not None:
            dt = t - prev_pt["time"]
            if dt <= 0:
                dt = 1e-3  # strictly enforce positive delta
            dist = haversine_distance(prev_pt["raw_lon"], prev_pt["raw_lat"], lon, lat)
            speed_mps = dist / dt if dt > 0 else 0.0
            speed_kmh = speed_mps * 3.6
            bearing = calculate_bearing(prev_pt["raw_lon"], prev_pt["raw_lat"], lon, lat)
            
            audit["max_raw_speed_kmh"] = max(audit["max_raw_speed_kmh"], speed_kmh)
            
            # Check for coordinate duplicate
            if dist < 0.1:  # less than 10 cm
                is_duplicate = True
                audit["duplicate_coords_handled"] += 1
                bearing = prev_pt["bearing"]  # maintain previous heading
                
            # Check for abnormal jump / speed outlier
            if speed_kmh > speed_limit_kmh:
                is_speed_outlier = True
                audit["speed_outliers_flagged"] += 1
                
            # Check for temporal gap (> 10s)
            if dt > 10.0:
                is_temporal_gap = True
                audit["temporal_gaps_flagged"] += 1
        else:
            # First point
            if len(raw_points) > 1:
                bearing = calculate_bearing(lon, lat, raw_points[1]["lon"], raw_points[1]["lat"])
            else:
                bearing = 0.0
                
        # Apply Kalman smoothing
        smooth_lon, smooth_lat = kf.step(lon, lat, dt)
        
        cleaned_pt = {
            "point_idx": idx,
            "lon": smooth_lon,
            "lat": smooth_lat,
            "raw_lon": lon,
            "raw_lat": lat,
            "time": t,
            "dt": dt,
            "dist_from_prev": dist,
            "speed_mps": speed_mps,
            "speed_kmh": speed_kmh,
            "bearing": bearing,
            "smooth_lon": smooth_lon,
            "smooth_lat": smooth_lat,
            "is_duplicate": is_duplicate,
            "is_speed_outlier": is_speed_outlier,
            "is_temporal_gap": is_temporal_gap
        }
        
        if prev_pt is not None and not is_speed_outlier:
            audit["total_distance_meters"] += dist
            audit["max_cleaned_speed_kmh"] = max(audit["max_cleaned_speed_kmh"], speed_kmh)
            
        cleaned.append(cleaned_pt)
        prev_pt = cleaned_pt
        
    if cleaned:
        audit["total_duration_seconds"] = cleaned[-1]["time"] - cleaned[0]["time"]
    audit["num_cleaned"] = len(cleaned)
    return cleaned, audit
