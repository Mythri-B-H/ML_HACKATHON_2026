"""
PathSnap - Dataset Manager & Pipeline Orchestrator
Manages tiered storage (raw, interim, processed, samples, reports)
and handles leak-free train/validation/test splits.
"""

import os
import glob
import json
import shutil
from typing import List, Dict, Any, Tuple
from backend.src.data.parser import load_trajectory_bundle
from backend.src.data.cleaner import clean_trajectory

class DatasetManager:
    def __init__(self, root_data_dir: str):
        self.root_data_dir = root_data_dir
        self.raw_dir = os.path.join(root_data_dir, "raw")
        self.interim_dir = os.path.join(root_data_dir, "interim")
        self.processed_dir = os.path.join(root_data_dir, "processed")
        self.samples_dir = os.path.join(root_data_dir, "samples")
        self.reports_dir = os.path.join(root_data_dir, "reports")
        
        for d in [self.interim_dir, self.processed_dir, self.samples_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)
            
    def get_trajectory_ids(self, split: str = "train") -> List[str]:
        split_dir = os.path.join(self.raw_dir, split)
        if not os.path.exists(split_dir):
            return []
        traj_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
        return [os.path.basename(d) for d in traj_dirs if os.path.isdir(d)]
        
    def get_train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        Splits the 80 training trajectories into 64 train and 16 validation trajectories
        using deterministic hashing / seeded shuffle to guarantee reproducibility.
        """
        train_ids = self.get_trajectory_ids("train")
        import random
        rng = random.Random(seed)
        shuffled = list(train_ids)
        rng.shuffle(shuffled)
        
        n_val = int(len(shuffled) * val_ratio)
        val_split = sorted(shuffled[:n_val])
        train_split = sorted(shuffled[n_val:])
        return train_split, val_split
        
    def create_samples(self, sample_traj_ids: List[str] = None):
        """
        Copies small representative trajectories into data/samples/ for fast testing.
        """
        if sample_traj_ids is None:
            sample_traj_ids = ["00000001", "00000002"]
            
        for tid in sample_traj_ids:
            src_dir = os.path.join(self.raw_dir, "train", tid)
            dst_dir = os.path.join(self.samples_dir, tid)
            if os.path.exists(src_dir) and not os.path.exists(dst_dir):
                shutil.copytree(src_dir, dst_dir)
                
    def process_and_cache_trajectory(self, split: str, traj_id: str) -> Dict[str, Any]:
        """
        Parses raw trajectory, cleans points, calculates kinematics,
        and saves processed summary.
        """
        traj_dir = os.path.join(self.raw_dir, split, traj_id)
        bundle = load_trajectory_bundle(traj_dir, traj_id)
        
        cleaned_points, audit = clean_trajectory(bundle["track"])
        
        # Save interim json representation
        out_split_dir = os.path.join(self.processed_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        processed_data = {
            "traj_id": traj_id,
            "split": split,
            "audit": audit,
            "cleaned_track": cleaned_points,
            "num_nodes": len(bundle["nodes"]),
            "num_arcs": len(bundle["arcs"]),
            "route": bundle["route"]
        }
        
        out_path = os.path.join(out_split_dir, f"{traj_id}_processed.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f)
            
        return processed_data

    def run_cleaning_pipeline(self) -> Dict[str, Any]:
        """
        Runs cleaning across all training and student test trajectories,
        logging audit reports to data/reports/cleaning_summary.json.
        """
        overall_audit = {
            "train": {},
            "student_test": {},
            "totals": {
                "total_points_cleaned": 0,
                "total_duplicates_handled": 0,
                "total_speed_outliers_flagged": 0,
                "total_temporal_gaps_flagged": 0
            }
        }
        
        # Create samples first
        self.create_samples(["00000001", "00000002"])
        
        for split in ["train", "student_test"]:
            traj_ids = self.get_trajectory_ids(split)
            for tid in traj_ids:
                data = self.process_and_cache_trajectory(split, tid)
                audit = data["audit"]
                overall_audit[split][tid] = audit
                overall_audit["totals"]["total_points_cleaned"] += audit["num_cleaned"]
                overall_audit["totals"]["total_duplicates_handled"] += audit["duplicate_coords_handled"]
                overall_audit["totals"]["total_speed_outliers_flagged"] += audit["speed_outliers_flagged"]
                overall_audit["totals"]["total_temporal_gaps_flagged"] += audit["temporal_gaps_flagged"]
                
        report_path = os.path.join(self.reports_dir, "cleaning_summary.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(overall_audit, f, indent=2)
            
        return overall_audit
