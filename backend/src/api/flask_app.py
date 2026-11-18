"""
PathSnap - Backend REST API Service
Flask-based high-performance API providing trajectory upload, validation,
ML inference, HMM/Viterbi route reconstruction, model benchmarks, and frontend static serving.
"""

import os
import glob
import json
import time
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory, send_file

from backend.src.data.parser import load_trajectory_bundle
from backend.src.data.cleaner import clean_trajectory
from backend.src.network.graph import RoadNetwork
from backend.src.network.spatial_index import SpatialGridIndex
from backend.src.ml.models import get_all_models
from backend.src.hmm.pipeline import RouteReconstructionPipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "saved")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)

# CORS Header support
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# In-memory graph & spatial caches
NETWORK_CACHE: Dict[str, RoadNetwork] = {}
SPATIAL_CACHE: Dict[str, SpatialGridIndex] = {}
LOADED_MODELS: Dict[str, Any] = {}

def get_loaded_model(model_key: str):
    if not model_key or model_key in ["none", "pure_hmm", "none (pure hmm)"]:
        return None
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]
    model_path = os.path.join(MODELS_DIR, f"{model_key}.joblib")
    if os.path.exists(model_path):
        all_models = get_all_models()
        if model_key in all_models:
            m = all_models[model_key]
            m.load(model_path)
            LOADED_MODELS[model_key] = m
            return m
    return None

@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "PathSnap Intelligent GPS Route Reconstruction",
        "version": "1.0.0"
    })

@app.route("/api/trajectories", methods=["GET"])
def list_trajectories():
    res = {}
    for split in ["train", "student_test", "custom"]:
        sdir = os.path.join(DATA_DIR, "raw", split)
        if os.path.exists(sdir):
            trajs = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(sdir, "*"))) if os.path.isdir(d)]
            res[split] = trajs
        else:
            res[split] = []
    return jsonify({
        "status": "success",
        "splits": res,
        "total_trajectories": sum(len(v) for v in res.values())
    })

@app.route("/api/trajectory/<split>/<traj_id>", methods=["GET"])
def get_trajectory_details(split, traj_id):
    tdir = os.path.join(DATA_DIR, "raw", split, traj_id)
    if not os.path.exists(tdir):
        return jsonify({"status": "error", "message": f"Trajectory not found: {traj_id}"}), 404
        
    bundle = load_trajectory_bundle(tdir, traj_id)
    cleaned, audit = clean_trajectory(bundle["track"])
    
    lons = [pt["lon"] for pt in bundle["track"]]
    lats = [pt["lat"] for pt in bundle["track"]]
    bbox = [min(lons), min(lats), max(lons), max(lats)] if lons else [0, 0, 0, 0]
    
    preview_step = max(1, len(bundle["track"]) // 300)
    preview_points = [[pt["lon"], pt["lat"], pt["time"]] for pt in bundle["track"][::preview_step]]
    
    return jsonify({
        "status": "success",
        "traj_id": traj_id,
        "split": split,
        "num_track_points": len(bundle["track"]),
        "num_nodes": len(bundle["nodes"]),
        "num_arcs": len(bundle["arcs"]),
        "num_route_arcs": len(bundle["route"]),
        "bbox": bbox,
        "audit": audit,
        "preview_track": preview_points
    })

@app.route("/api/reconstruct", methods=["POST"])
def reconstruct_route():
    body = request.get_json(silent=True) or {}
    split = body.get("split", "train")
    traj_id = body.get("traj_id", "00000001")
    model_key = body.get("model", "random_forest")
    search_radius_m = float(body.get("search_radius_m", 60.0))
    max_gps_points = int(body.get("max_gps_points", 1500))
    
    tdir = os.path.join(DATA_DIR, "raw", split, traj_id)
    if not os.path.exists(tdir):
        return jsonify({"status": "error", "message": f"Trajectory {traj_id} not found in {split}"}), 404
        
    bundle = load_trajectory_bundle(tdir, traj_id)
    
    cache_key = f"{split}_{traj_id}"
    if cache_key not in NETWORK_CACHE:
        net = RoadNetwork(bundle["nodes"], bundle["arcs"])
        idx = SpatialGridIndex(net)
        NETWORK_CACHE[cache_key] = net
        SPATIAL_CACHE[cache_key] = idx
    else:
        net = NETWORK_CACHE[cache_key]
        idx = SPATIAL_CACHE[cache_key]
        
    ml_model = get_loaded_model(model_key)
    pipeline = RouteReconstructionPipeline(
        net, spatial_index=idx, ml_model=ml_model, search_radius_m=search_radius_m
    )
    
    raw_track = bundle["track"]
    if len(raw_track) > max_gps_points:
        step = max(1, len(raw_track) // max_gps_points)
        raw_track = raw_track[::step]
        
    res = pipeline.reconstruct(raw_track, bundle["route"])
    raw_coords = [[pt["lon"], pt["lat"]] for pt in raw_track]
    
    return jsonify({
        "status": "success",
        "traj_id": traj_id,
        "split": split,
        "model_used": res["used_ml_model"],
        "processing_time_ms": res["processing_time_ms"],
        "reconstructed_distance_km": res["reconstructed_distance_km"],
        "num_gps_raw": res["num_gps_raw"],
        "num_gps_cleaned": res["num_gps_cleaned"],
        "num_gaps_bridged": res["num_gaps_bridged"],
        "continuous_arcs": res["continuous_arcs"],
        "raw_gps_coordinates": raw_coords,
        "reconstructed_coordinates": res["coordinates"],
        "evaluation": res["evaluation"],
        "ml_only_evaluation": res["ml_only_evaluation"],
        "cleaning_audit": res["cleaning_audit"]
    })

@app.route("/api/models", methods=["GET"])
def get_models_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    return jsonify({"status": "success", "models": metrics})

@app.route("/api/upload", methods=["POST"])
def upload_trajectory():
    try:
        traj_id = request.form.get("traj_id") or f"custom_{int(time.time())}"
        target_dir = os.path.join(DATA_DIR, "raw", "custom", traj_id)
        os.makedirs(target_dir, exist_ok=True)
        
        saved_files = []
        for ext in ["track", "nodes", "arcs", "route"]:
            if ext in request.files:
                f = request.files[ext]
                out_path = os.path.join(target_dir, f"{traj_id}.{ext}")
                f.save(out_path)
                saved_files.append(ext)
                
        has_track = os.path.exists(os.path.join(target_dir, f"{traj_id}.track"))
        has_nodes = os.path.exists(os.path.join(target_dir, f"{traj_id}.nodes"))
        has_arcs = os.path.exists(os.path.join(target_dir, f"{traj_id}.arcs"))
        
        if not (has_track and has_nodes and has_arcs):
            return jsonify({
                "status": "error",
                "message": "Upload missing required files (.track, .nodes, .arcs)",
                "saved_files": saved_files
            }), 400
            
        bundle = load_trajectory_bundle(target_dir, traj_id)
        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded and validated trajectory {traj_id}",
            "traj_id": traj_id,
            "split": "custom",
            "num_track_points": len(bundle["track"]),
            "num_nodes": len(bundle["nodes"]),
            "num_arcs": len(bundle["arcs"]),
            "has_route": len(bundle["route"]) > 0
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/validate", methods=["POST"])
def validate_trajectory():
    body = request.get_json(silent=True) or {}
    split = body.get("split", "train")
    traj_id = body.get("traj_id", "00000001")
    tdir = os.path.join(DATA_DIR, "raw", split, traj_id)
    if not os.path.exists(tdir):
        return jsonify({"status": "error", "message": f"Trajectory {traj_id} not found"}), 404
        
    bundle = load_trajectory_bundle(tdir, traj_id)
    cleaned, audit = clean_trajectory(bundle["track"])
    
    num_nodes = len(bundle["nodes"])
    num_arcs = len(bundle["arcs"])
    invalid_arcs = sum(1 for u, v in bundle["arcs"] if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes)
    invalid_routes = sum(1 for a in bundle["route"] if a < 0 or a >= num_arcs)
    
    return jsonify({
        "status": "success",
        "traj_id": traj_id,
        "split": split,
        "is_valid": (invalid_arcs == 0 and invalid_routes == 0),
        "num_track_points": len(bundle["track"]),
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "invalid_arc_references": invalid_arcs,
        "invalid_route_references": invalid_routes,
        "cleaning_audit": audit
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
