"""
Unit Tests for Data Cleaning & Kinematics
"""

import pytest
from backend.src.data.cleaner import clean_trajectory, KalmanFilter2D

def test_kalman_filter():
    kf = KalmanFilter2D()
    lon1, lat1 = kf.step(8.59, 49.84, 1.0)
    assert abs(lon1 - 8.59) < 1e-4
    assert abs(lat1 - 49.84) < 1e-4
    
    # Step 2 with small increment
    lon2, lat2 = kf.step(8.5901, 49.8401, 1.0)
    assert lon2 > lon1

def test_clean_trajectory_kinematics():
    mock_points = [
        {"lon": 8.5900, "lat": 49.8400, "time": 1.0},
        {"lon": 8.5901, "lat": 49.8400, "time": 2.0},  # ~7 meters in 1s (~25 km/h)
        {"lon": 8.5901, "lat": 49.8400, "time": 3.0},  # duplicate coord (stationary)
        {"lon": 8.6500, "lat": 49.8400, "time": 4.0},  # huge jump (~4 km in 1s -> speed outlier)
    ]
    
    cleaned, audit = clean_trajectory(mock_points, speed_limit_kmh=150.0)
    assert len(cleaned) == 4
    assert audit["num_cleaned"] == 4
    assert audit["duplicate_coords_handled"] == 1
    assert audit["speed_outliers_flagged"] == 1
    assert cleaned[2]["is_duplicate"] is True
    assert cleaned[3]["is_speed_outlier"] is True
