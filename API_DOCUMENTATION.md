# PathSnap – REST API Documentation

The PathSnap REST API provides programmatic access to dataset inspection, trajectory validation, ML candidate scoring, and HMM/Viterbi route reconstruction.

Base URL: `http://127.0.0.1:8000`

---

## 1. System Health

### `GET /api/health`
Checks server status and operational readiness.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "PathSnap Intelligent GPS Route Reconstruction",
  "version": "1.0.0"
}
```

---

## 2. Dataset Management

### `GET /api/trajectories`
Lists all available trajectories grouped by dataset split (`train`, `student_test`, `custom`).

**Response:** `200 OK`
```json
{
  "status": "success",
  "splits": {
    "train": ["00000001", "00000002", "..."],
    "student_test": ["00000000", "00000007", "..."],
    "custom": []
  },
  "total_trajectories": 90
}
```

---

### `GET /api/trajectory/{split}/{traj_id}`
Returns metadata, bounding box, kinematics audit, and downsampled track points for a specific trajectory.

**Parameters:**
- `split` (string, path): `train` | `student_test` | `custom`
- `traj_id` (string, path): e.g. `00000001`

**Response:** `200 OK`
```json
{
  "status": "success",
  "traj_id": "00000001",
  "split": "train",
  "num_track_points": 1315,
  "num_nodes": 88809,
  "num_arcs": 165790,
  "num_route_arcs": 525,
  "bbox": [8.594, 49.839, 8.598, 49.855],
  "audit": {
    "num_raw": 1315,
    "num_cleaned": 1315,
    "duplicate_coords_handled": 0,
    "speed_outliers_flagged": 5
  },
  "preview_track": [[8.595203, 49.844053, 1.0], "..."]
}
```

---

## 3. Route Reconstruction

### `POST /api/reconstruct`
Executes map-matching and sequence reconstruction on a trajectory.

**Request Body (JSON):**
```json
{
  "split": "train",
  "traj_id": "00000001",
  "model": "random_forest",
  "search_radius_m": 60.0,
  "max_gps_points": 1500
}
```

**Fields:**
- `split`: Dataset partition (`train`, `student_test`, `custom`).
- `traj_id`: Trajectory ID string.
- `model`: One of `random_forest`, `gradient_boosting`, `mlp`, `svm`, `logistic_regression`, or `pure_hmm`.
- `search_radius_m`: Spatial candidate search radius in meters (default: `60.0`).
- `max_gps_points`: Maximum GPS points to process for high throughput (default: `1500`).

**Response:** `200 OK`
```json
{
  "status": "success",
  "traj_id": "00000001",
  "split": "train",
  "model_used": "Random Forest",
  "processing_time_ms": 1150.2,
  "reconstructed_distance_km": 53.85,
  "num_gps_raw": 1315,
  "num_gps_cleaned": 1315,
  "num_gaps_bridged": 209,
  "continuous_arcs": [2641, 64973, 133948, 133949, "..."],
  "raw_gps_coordinates": [[8.595203, 49.844053], "..."],
  "reconstructed_coordinates": [[8.594417, 49.839972], [8.595156, 49.843765], "..."],
  "evaluation": {
    "precision": 0.7676,
    "recall": 0.9752,
    "f1_score": 0.8591,
    "jaccard_similarity": 0.7529
  },
  "ml_only_evaluation": {
    "precision": 0.6905,
    "recall": 0.7352,
    "f1_score": 0.7122
  }
}
```

---

## 4. Model Benchmarks

### `GET /api/models`
Returns trained performance metrics, confusion matrices, and training times for all 5 models.

**Response:** `200 OK`
```json
{
  "status": "success",
  "models": {
    "logistic_regression": { "name": "Logistic Regression", "accuracy": 0.7614, "f1_score": 0.696, "roc_auc": 0.8274 },
    "random_forest": { "name": "Random Forest", "accuracy": 0.7786, "f1_score": 0.7015, "roc_auc": 0.8348 },
    "gradient_boosting": { "name": "HistGradientBoosting", "accuracy": 0.7769, "f1_score": 0.705, "roc_auc": 0.8388 },
    "mlp": { "name": "Multi-Layer Perceptron", "accuracy": 0.8108, "f1_score": 0.7114, "roc_auc": 0.8500 },
    "svm": { "name": "Support Vector Machine", "accuracy": 0.786, "f1_score": 0.691, "roc_auc": 0.8259 }
  }
}
```

---

## 5. Upload & Validation

### `POST /api/upload`
Uploads raw trajectory files as multipart form data.

**Form Data:**
- `traj_id`: Identifier for the uploaded trajectory.
- `track`: `.track` file (required).
- `nodes`: `.nodes` file (required).
- `arcs`: `.arcs` file (required).
- `route`: `.route` file (optional ground truth).

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Successfully uploaded and validated trajectory custom_trip_01",
  "traj_id": "custom_trip_01",
  "num_track_points": 450,
  "num_nodes": 3200,
  "num_arcs": 6400,
  "has_route": true
}
```

---

### `POST /api/validate`
Validates structural integrity and reference bounds of any trajectory.

**Request Body (JSON):**
```json
{
  "split": "train",
  "traj_id": "00000001"
}
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "traj_id": "00000001",
  "is_valid": true,
  "invalid_arc_references": 0,
  "invalid_route_references": 0
}
```
