"""
Calculate per-joint error statistics from predictions and ground truth.

This script:
1. Loads predictions and ground truth annotations
2. Calculates per-joint errors (pixel distance, detection rate)
3. Saves detailed per-joint statistics to CSV
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .common.coco_schema import COCO_KEYPOINT_NAMES


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def calculate_per_joint_errors(
    predictions_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
    min_conf: float = 0.3,
) -> Dict:
    """
    Calculate per-joint error statistics.
    
    Args:
        predictions_file: Path to predictions JSON
        ground_truth_file: Path to ground truth JSON
        output_dir: Directory to save results
        min_conf: Minimum confidence threshold for joint detection
        
    Returns:
        Dictionary with per-joint statistics
    """
    logger = setup_logging()
    
    logger.info(f"Calculating per-joint errors")
    logger.info(f"Predictions: {predictions_file}")
    logger.info(f"Ground truth: {ground_truth_file}")
    
    # Load predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    
    # Load ground truth
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)
    
    if not predictions or not ground_truth:
        logger.error("No predictions or ground truth found!")
        return {"success": False, "error": "Missing data"}
    
    # Create mapping from image_name to ground truth
    gt_map = {gt["image_name"]: gt for gt in ground_truth}
    
    model_name = predictions_file.stem.replace("_predictions", "")
    logger.info(f"Model: {model_name}")
    
    # Initialize per-joint statistics
    n_joints = 17
    joint_errors = [[] for _ in range(n_joints)]
    joint_detected = [0] * n_joints
    joint_visible = [0] * n_joints
    joint_total = [0] * n_joints
    
    # Calculate errors for each prediction
    matched_predictions = 0
    for pred in predictions:
        image_name = pred["image_name"]
        
        if image_name not in gt_map:
            continue
        
        matched_predictions += 1
        gt = gt_map[image_name]
        
        pred_keypoints = np.array(pred["keypoints"])  # (17, 2)
        pred_confidences = np.array(pred["confidences"])  # (17,)
        gt_keypoints = np.array(gt["keypoints"])  # (17, 2)
        gt_visible = np.array(gt["visible"])  # (17,)
        
        # Calculate per-joint errors
        for joint_idx in range(n_joints):
            joint_total[joint_idx] += 1
            
            # Check if joint is visible in ground truth
            if gt_visible[joint_idx]:
                joint_visible[joint_idx] += 1
                
                # Check if joint is detected by model
                if pred_confidences[joint_idx] >= min_conf:
                    joint_detected[joint_idx] += 1
                    
                    # Calculate pixel error
                    pred_pt = pred_keypoints[joint_idx]
                    gt_pt = gt_keypoints[joint_idx]
                    error = np.linalg.norm(pred_pt - gt_pt)
                    joint_errors[joint_idx].append(error)
    
    logger.info(f"Matched predictions: {matched_predictions}/{len(predictions)}")
    
    # Calculate statistics for each joint
    per_joint_stats = []
    
    for joint_idx in range(n_joints):
        joint_name = COCO_KEYPOINT_NAMES[joint_idx]
        
        # Detection rate (of visible joints)
        detection_rate = (joint_detected[joint_idx] / joint_visible[joint_idx] * 100
                         if joint_visible[joint_idx] > 0 else 0.0)
        
        # Error statistics (only for detected joints)
        errors = joint_errors[joint_idx]
        mean_error = float(np.mean(errors)) if errors else np.nan
        std_error = float(np.std(errors)) if errors else np.nan
        median_error = float(np.median(errors)) if errors else np.nan
        
        stats = {
            "joint_idx": joint_idx,
            "joint_name": joint_name,
            "total_images": joint_total[joint_idx],
            "visible_count": joint_visible[joint_idx],
            "detected_count": joint_detected[joint_idx],
            "detection_rate": float(detection_rate),
            "mean_error_px": mean_error,
            "std_error_px": std_error,
            "median_error_px": median_error,
            "num_errors": len(errors),
        }
        
        per_joint_stats.append(stats)
    
    # Create DataFrame
    df = pd.DataFrame(per_joint_stats)
    
    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_per_joint_errors.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved per-joint errors to {output_file}")
    
    # Print summary
    logger.info(f"\nPer-Joint Error Summary:")
    logger.info(f"\n{df.to_string(index=False)}")
    
    # Calculate overall statistics
    overall_stats = {
        "model_name": model_name,
        "matched_predictions": matched_predictions,
        "mean_detection_rate": float(df["detection_rate"].mean()),
        "mean_pixel_error": float(df["mean_error_px"].mean()),
        "std_pixel_error": float(df["std_error_px"].mean()),
    }
    
    return {
        "success": True,
        "per_joint_file": str(output_file),
        "overall_stats": overall_stats,
    }


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Calculate per-joint error statistics"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save per-joint error statistics",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for joint detection",
    )
    
    args = parser.parse_args()
    
    result = calculate_per_joint_errors(
        predictions_file=args.predictions,
        ground_truth_file=args.ground_truth,
        output_dir=args.output_dir,
        min_conf=args.min_conf,
    )
    
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
