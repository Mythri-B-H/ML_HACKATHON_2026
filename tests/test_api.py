"""
Integration Tests for REST API Endpoints (Flask Client)
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pytest
from backend.src.api.flask_app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"

def test_api_trajectories(client):
    res = client.get("/api/trajectories")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["splits"]["train"]) == 80
    assert len(data["splits"]["student_test"]) == 10

def test_api_models(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert len(data["models"]) == 5

def test_api_validate(client):
    payload = {"split": "train", "traj_id": "00000001"}
    res = client.post("/api/validate", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["is_valid"] is True

def test_api_reconstruct(client):
    payload = {
        "split": "train",
        "traj_id": "00000001",
        "model": "random_forest",
        "search_radius_m": 60.0,
        "max_gps_points": 50
    }
    res = client.post("/api/reconstruct", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["reconstructed_coordinates"]) > 0
    assert "continuous_arcs" in data

def test_api_serve_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"PathSnap" in res.data
