"""
PathSnap - ML Training & Evaluation Pipeline
Trains the 5 models on 64 training trajectories, validates on 16 validation trajectories,
benchmarks performance metrics, and serializes trained models.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from backend.src.data.parser import load_trajectory_bundle
from backend.src.data.dataset import DatasetManager
from backend.src.ml.features import build_training_dataset
from backend.src.ml.models import get_all_models

def train_and_evaluate_all(data_root: str, models_save_dir: str) -> Dict[str, Any]:
    dm = DatasetManager(data_root)
    train_ids, val_ids = dm.get_train_val_split(val_ratio=0.2, seed=42)
    
    print(f"Loading {len(train_ids)} training trajectories...")
    train_bundles = []
    for tid in train_ids:
        tdir = os.path.join(dm.raw_dir, "train", tid)
        train_bundles.append(load_trajectory_bundle(tdir, tid))
        
    print(f"Loading {len(val_ids)} validation trajectories...")
    val_bundles = []
    for tid in val_ids:
        tdir = os.path.join(dm.raw_dir, "train", tid)
        val_bundles.append(load_trajectory_bundle(tdir, tid))
        
    print("Extracting features from training trajectories...")
    X_train, y_train = build_training_dataset(train_bundles, max_samples_per_traj=250)
    print(f"X_train shape: {X_train.shape}, positive samples: {np.sum(y_train)} / {len(y_train)}")
    
    print("Extracting features from validation trajectories...")
    X_val, y_val = build_training_dataset(val_bundles, max_samples_per_traj=250)
    print(f"X_val shape: {X_val.shape}, positive samples: {np.sum(y_val)} / {len(y_val)}")
    
    models = get_all_models()
    metrics_summary = {}
    os.makedirs(models_save_dir, exist_ok=True)
    
    for key, model in models.items():
        print(f"\nTraining model: {model.name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        probs = model.predict_proba(X_val)
        preds = (probs[:, 1] >= 0.5).astype(int)
        infer_time = (time.time() - t0) * 1000.0  # ms
        
        acc = float(accuracy_score(y_val, preds))
        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))
        f1 = float(f1_score(y_val, preds, zero_division=0))
        auc = float(roc_auc_score(y_val, probs[:, 1]))
        cm = confusion_matrix(y_val, preds).tolist()
        
        # Save model weights
        save_path = os.path.join(models_save_dir, f"{key}.joblib")
        model.save(save_path)
        
        metrics_summary[key] = {
            "name": model.name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "train_time_sec": round(train_time, 2),
            "inference_time_ms": round(infer_time, 2),
            "model_path": save_path
        }
        print(f"[{model.name}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    metrics_file = os.path.join(models_save_dir, "metrics_comparison.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"\nAll models saved and metrics recorded to: {metrics_file}")
    return metrics_summary

if __name__ == "__main__":
    data_dir = r"c:\Users\bhmyt\PathSnap\data"
    save_dir = r"c:\Users\bhmyt\PathSnap\models\saved"
    train_and_evaluate_all(data_dir, save_dir)
