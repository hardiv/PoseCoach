"""
Calculate benchmark metrics from prediction files.

This script:
1. Loads predictions from JSON files
2. Calculates aggregate metrics (detection rates, confidence scores, etc.)
3. Saves metrics to CSV files
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


def calculate_metrics(
    predictions_file: Path,
    ground_truth_file: Optional[Path],
    output_dir: Path,
    min_conf: float = 0.3,
) -> Dict:
    """
    Calculate benchmark metrics from predictions.
    
    Args:
        predictions_file: Path to predictions JSON
        ground_truth_file: Path to ground truth JSON (optional)
        output_dir: Directory to save metrics
        min_conf: Minimum confidence threshold for joint detection
        
    Returns:
        Dictionary with calculated metrics
    """
    logger = setup_logging()
    
    logger.info(f"Calculating metrics from {predictions_file}")
    logger.info(f"Confidence threshold: {min_conf}")
    
    # Load predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    
    if not predictions:
        logger.error("No predictions found!")
        return {"success": False, "error": "No predictions"}
    
    model_name = predictions_file.stem.replace("_predictions", "")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Load ground truth if available
    ground_truth = None
    if ground_truth_file and ground_truth_file.exists():
        with open(ground_truth_file, "r") as f:
            ground_truth = json.load(f)
        logger.info(f"Loaded ground truth: {len(ground_truth)} annotations")
    
    # Calculate per-image metrics
    per_image_metrics = []
    
    for pred in predictions:
        keypoints = np.array(pred["keypoints"])  # (17, 2)
        confidences = np.array(pred["confidences"])  # (17,)
        
        # Detection statistics
        detected_joints = np.sum(confidences >= min_conf)
        mean_conf_all = np.mean(confidences)
        mean_conf_detected = np.mean(confidences[confidences >= min_conf]) if detected_joints > 0 else 0.0
        valid_pose = detected_joints >= 8  # At least 8 joints for valid pose
        
        metrics = {
            "image_name": pred["image_name"],
            "model_name": model_name,
            "detected_joints": int(detected_joints),
            "mean_conf_all": float(mean_conf_all),
            "mean_conf_detected": float(mean_conf_detected),
            "valid_pose": bool(valid_pose),
            "inference_time_ms": pred.get("inference_time_ms", 0.0),
        }
        
        per_image_metrics.append(metrics)
    
    # Calculate aggregate metrics
    df = pd.DataFrame(per_image_metrics)
    
    aggregate_metrics = {
        "model_name": model_name,
        "num_images": len(predictions),
        "pose_detection_rate": float(df["valid_pose"].mean() * 100),  # %
        "mean_detected_joints": float(df["detected_joints"].mean()),
        "mean_confidence": float(df["mean_conf_all"].mean()),
        "mean_inference_time_ms": float(df["inference_time_ms"].mean()),
        "std_inference_time_ms": float(df["inference_time_ms"].std()),
    }
    
    # Save per-image metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_file = output_dir / f"{model_name}_per_image_metrics.csv"
    df.to_csv(per_image_file, index=False)
    logger.info(f"Saved per-image metrics to {per_image_file}")
    
    # Save aggregate metrics
    aggregate_file = output_dir / f"{model_name}_aggregate_metrics.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    logger.info(f"Saved aggregate metrics to {aggregate_file}")
    
    # Log summary
    logger.info(f"\nMetrics Summary:")
    logger.info(f"  Pose detection rate: {aggregate_metrics['pose_detection_rate']:.1f}%")
    logger.info(f"  Mean detected joints: {aggregate_metrics['mean_detected_joints']:.1f}")
    logger.info(f"  Mean confidence: {aggregate_metrics['mean_confidence']:.3f}")
    logger.info(f"  Mean inference time: {aggregate_metrics['mean_inference_time_ms']:.2f} ms")
    
    return {
        "success": True,
        "aggregate_metrics": aggregate_metrics,
        "per_image_file": str(per_image_file),
        "aggregate_file": str(aggregate_file),
    }


def create_leaderboard(metrics_dir: Path, output_dir: Path) -> None:
    """
    Create leaderboard from multiple aggregate metrics files.
    
    Args:
        metrics_dir: Directory containing aggregate metrics JSON files
        output_dir: Directory to save leaderboard
    """
    logger = setup_logging()
    
    logger.info(f"Creating leaderboard from {metrics_dir}")
    
    # Find all aggregate metrics files
    metrics_files = list(metrics_dir.glob("*_aggregate_metrics.json"))
    
    if not metrics_files:
        logger.warning("No aggregate metrics files found!")
        return
    
    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    # Create leaderboard DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by pose detection rate (descending)
    df = df.sort_values("pose_detection_rate", ascending=False)
    
    # Save leaderboard
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_file = output_dir / "leaderboard.csv"
    df.to_csv(leaderboard_file, index=False)
    logger.info(f"Saved leaderboard to {leaderboard_file}")
    
    # Print leaderboard
    logger.info(f"\n{'='*80}")
    logger.info("LEADERBOARD")
    logger.info(f"{'='*80}")
    print("\n" + df.to_string(index=False))
    print()


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Calculate benchmark metrics from predictions"
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
        default=None,
        help="Path to ground truth JSON file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save metrics",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for joint detection",
    )
    parser.add_argument(
        "--create-leaderboard",
        action="store_true",
        help="Create leaderboard from all metrics in output-dir",
    )
    
    args = parser.parse_args()
    
    if args.create_leaderboard:
        create_leaderboard(args.output_dir, args.output_dir)
    else:
        result = calculate_metrics(
            predictions_file=args.predictions,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir,
            min_conf=args.min_conf,
        )
        
        if not result["success"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
