"""Metrics computation for pose estimation benchmarking."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


class PoseMetrics:
    """Compute and store pose estimation metrics."""
    
    def __init__(self, min_conf: float = 0.2):
        """
        Initialize metrics tracker.
        
        Args:
            min_conf: Minimum confidence threshold for joint detection
        """
        self.min_conf = min_conf
        self.records: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        image_name: str,
        model_name: str,
        keypoints: np.ndarray,
        conf: np.ndarray,
        gt_keypoints: np.ndarray | None = None,
        gt_visible: np.ndarray | None = None,
        inference_time_ms: float | None = None,
    ) -> None:
        """
        Add a pose estimation result and compute metrics.
        
        Args:
            image_name: Name of the image file
            model_name: Name of the pose model
            keypoints: Joint coordinates (17, 2)
            conf: Joint confidences (17,)
            gt_keypoints: Ground truth keypoints (17, 2) - optional, for MPII
            gt_visible: Ground truth visibility (17,) - optional, for MPII
            inference_time_ms: Inference time in milliseconds - optional
        """
        # Count detected joints (above confidence threshold)
        detected_mask = conf >= self.min_conf
        detected_joints_count = detected_mask.sum()
        
        # Mean confidence (over all joints)
        mean_conf_all = conf.mean()
        
        # Mean confidence (over detected joints only)
        if detected_joints_count > 0:
            mean_conf_detected = conf[detected_mask].mean()
        else:
            mean_conf_detected = 0.0
        
        # Valid pose: at least 8 joints detected (arbitrary threshold for "reasonable" pose)
        valid_pose = detected_joints_count >= 8
        
        # Compute pixel error if ground truth is provided (for MPII)
        mean_pixel_error = None
        if gt_keypoints is not None and gt_visible is not None:
            # Compute error only for visible ground truth joints
            visible_mask = (gt_visible > 0) & detected_mask
            if visible_mask.sum() > 0:
                errors = np.linalg.norm(
                    keypoints[visible_mask] - gt_keypoints[visible_mask],
                    axis=1
                )
                mean_pixel_error = float(errors.mean())
        
        record = {
            "image_name": image_name,
            "model_name": model_name,
            "detected_joints_count": int(detected_joints_count),
            "mean_conf_all": float(mean_conf_all),
            "mean_conf_detected": float(mean_conf_detected),
            "valid_pose": bool(valid_pose),
        }
        
        if mean_pixel_error is not None:
            record["mean_pixel_error"] = mean_pixel_error
        
        if inference_time_ms is not None:
            record["inference_time_ms"] = float(inference_time_ms)
        
        self.records.append(record)
    
    def save_per_image_metrics(self, output_dir: Path, model_name: str, dataset_name: str) -> None:
        """
        Save per-image metrics to CSV.
        
        Args:
            output_dir: Base output directory
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        metrics_dir = output_dir / "metrics" / dataset_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter records for this model
        model_records = [r for r in self.records if r["model_name"] == model_name]
        
        if not model_records:
            return
        
        df = pd.DataFrame(model_records)
        csv_path = metrics_dir / f"{model_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved per-image metrics to {csv_path}")
    
    def compute_leaderboard(self, output_dir: Path) -> pd.DataFrame:
        """
        Compute aggregate metrics across all models and save leaderboard.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            DataFrame with leaderboard results
        """
        if not self.records:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.records)
        
        # Base aggregation
        agg_dict = {
            "image_name": "count",  # num_images
            "valid_pose": "mean",   # pose_rate
            "detected_joints_count": "mean",
            "mean_conf_detected": "mean",
        }
        
        # Add pixel error if present (MPII)
        if "mean_pixel_error" in df.columns:
            agg_dict["mean_pixel_error"] = "mean"
        
        # Add timing statistics if present
        if "inference_time_ms" in df.columns:
            agg_dict["inference_time_ms"] = ["mean", "std"]
        
        # Aggregate by model
        leaderboard = df.groupby("model_name").agg(agg_dict).reset_index()
        
        # Flatten multi-level column names if present
        if hasattr(leaderboard.columns, 'levels'):  # MultiIndex check
            new_columns = []
            for col in leaderboard.columns:
                if isinstance(col, tuple):
                    # Flatten tuple columns
                    if col[1] == "":
                        new_columns.append(col[0])
                    else:
                        new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col)
            leaderboard.columns = new_columns
        
        # Rename columns - match the flattened names with suffixes
        rename_map = {
            "image_name_count": "num_images",
            "valid_pose_mean": "pose_rate",
            "detected_joints_count_mean": "mean_detected_joints",
            "mean_conf_detected_mean": "mean_conf",
            "mean_pixel_error_mean": "mean_pixel_error",
            "inference_time_ms_mean": "mean_inference_ms",
            "inference_time_ms_std": "std_inference_ms",
        }
        
        leaderboard = leaderboard.rename(columns=rename_map)
        
        # Convert pose_rate to percentage and round values
        leaderboard["pose_rate"] = (leaderboard["pose_rate"] * 100).round(2)
        leaderboard["mean_detected_joints"] = leaderboard["mean_detected_joints"].round(2)
        leaderboard["mean_conf"] = leaderboard["mean_conf"].round(3)
        
        if "mean_pixel_error" in leaderboard.columns:
            leaderboard["mean_pixel_error"] = leaderboard["mean_pixel_error"].round(2)
        
        if "mean_inference_ms" in leaderboard.columns:
            leaderboard["mean_inference_ms"] = leaderboard["mean_inference_ms"].round(2)
            leaderboard["std_inference_ms"] = leaderboard["std_inference_ms"].round(2)
        
        # Sort by pose_rate descending
        leaderboard = leaderboard.sort_values("pose_rate", ascending=False)
        
        # Save leaderboard
        leaderboard_path = output_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        print(f"\nSaved leaderboard to {leaderboard_path}")
        print("\nLeaderboard:")
        print(leaderboard.to_string(index=False))
        
        return leaderboard
