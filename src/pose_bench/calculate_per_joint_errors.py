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
from .common.error_metrics import calculate_pck, calculate_oks, calculate_normalized_error


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
    joint_errors_px = [[] for _ in range(n_joints)]
    joint_errors_norm = [[] for _ in range(n_joints)]
    joint_detected = [0] * n_joints
    joint_visible = [0] * n_joints
    joint_total = [0] * n_joints
    
    # Aggregate metrics
    all_pck_scores = []
    all_oks_scores = []
    all_normalized_errors = []
    
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
        
        # Calculate standardized metrics for this image
        # PCK@0.5 (MPII standard)
        pck_score, _ = calculate_pck(pred_keypoints, gt_keypoints, gt_visible, threshold=0.5)
        if not np.isnan(pck_score):
            all_pck_scores.append(pck_score)
        
        # OKS (COCO standard) - needs bbox for scale
        bbox = None  # Will use fallback scale calculation
        oks_score, _ = calculate_oks(pred_keypoints, pred_confidences, gt_keypoints, gt_visible, bbox)
        if not np.isnan(oks_score):
            all_oks_scores.append(oks_score)
        
        # Normalized error
        norm_error, per_joint_norm_error = calculate_normalized_error(pred_keypoints, gt_keypoints, gt_visible)
        if not np.isnan(norm_error):
            all_normalized_errors.append(norm_error)
        
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
                    error_px = np.linalg.norm(pred_pt - gt_pt)
                    joint_errors_px[joint_idx].append(error_px)
                    
                    # Store normalized error
                    if not np.isnan(per_joint_norm_error[joint_idx]):
                        joint_errors_norm[joint_idx].append(per_joint_norm_error[joint_idx])
    
    logger.info(f"Matched predictions: {matched_predictions}/{len(predictions)}")
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "pck_at_0.5": float(np.mean(all_pck_scores)) if all_pck_scores else np.nan,
        "oks": float(np.mean(all_oks_scores)) if all_oks_scores else np.nan,
        "mean_normalized_error": float(np.mean(all_normalized_errors)) if all_normalized_errors else np.nan,
    }
    
    logger.info(f"\nAggregate Error Metrics:")
    logger.info(f"  PCK@0.5: {aggregate_metrics['pck_at_0.5']:.3f}" if not np.isnan(aggregate_metrics['pck_at_0.5']) else "  PCK@0.5: N/A")
    logger.info(f"  OKS: {aggregate_metrics['oks']:.3f}" if not np.isnan(aggregate_metrics['oks']) else "  OKS: N/A")
    logger.info(f"  Normalized Error: {aggregate_metrics['mean_normalized_error']:.3f}" if not np.isnan(aggregate_metrics['mean_normalized_error']) else "  Normalized Error: N/A")
    
    # Calculate statistics for each joint
    per_joint_stats = []
    
    for joint_idx in range(n_joints):
        joint_name = COCO_KEYPOINT_NAMES[joint_idx]
        
        # Detection rate (of visible joints)
        detection_rate = (joint_detected[joint_idx] / joint_visible[joint_idx] * 100
                         if joint_visible[joint_idx] > 0 else 0.0)
        
        # Pixel error statistics (only for detected joints)
        errors_px = joint_errors_px[joint_idx]
        mean_error_px = float(np.mean(errors_px)) if errors_px else np.nan
        std_error_px = float(np.std(errors_px)) if errors_px else np.nan
        median_error_px = float(np.median(errors_px)) if errors_px else np.nan
        
        # Normalized error statistics
        errors_norm = joint_errors_norm[joint_idx]
        mean_error_norm = float(np.mean(errors_norm)) if errors_norm else np.nan
        median_error_norm = float(np.median(errors_norm)) if errors_norm else np.nan
        
        stats = {
            "joint_idx": joint_idx,
            "joint_name": joint_name,
            "total_images": joint_total[joint_idx],
            "visible_count": joint_visible[joint_idx],
            "detected_count": joint_detected[joint_idx],
            "detection_rate": float(detection_rate),
            "mean_error_px": mean_error_px,
            "std_error_px": std_error_px,
            "median_error_px": median_error_px,
            "mean_error_normalized": mean_error_norm,
            "median_error_normalized": median_error_norm,
            "num_errors": len(errors_px),
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
        "mean_normalized_error": float(df["mean_error_normalized"].mean()),
        "pck_at_0.5": aggregate_metrics["pck_at_0.5"],
        "oks": aggregate_metrics["oks"],
    }
    
    # Save aggregate metrics separately
    aggregate_file = output_dir / f"{model_name}_error_metrics.json"
    with open(aggregate_file, "w") as f:
        import json
        json.dump(aggregate_metrics, f, indent=2)
    logger.info(f"Saved aggregate error metrics to {aggregate_file}")
    
    return {
        "success": True,
        "per_joint_file": str(output_file),
        "aggregate_file": str(aggregate
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
