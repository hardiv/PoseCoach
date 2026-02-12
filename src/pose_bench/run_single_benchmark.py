"""
Run a complete benchmark for a single model on a single dataset.

This script orchestrates:
1. Inference - Run model predictions
2. Metrics - Calculate benchmark metrics
3. Per-joint errors - Calculate per-joint statistics (if ground truth available)
4. Overlays - Generate skeleton visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .inference import run_inference
from .calculate_metrics import calculate_metrics, create_leaderboard
from .calculate_per_joint_errors import calculate_per_joint_errors
from .generate_overlays import generate_overlays


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def run_single_benchmark(
    model_name: str,
    config: Config,
) -> bool:
    """
    Run complete benchmark for one model.
    
    Args:
        model_name: Name of the model to benchmark
        config: Benchmark configuration
        
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logging()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running benchmark: {model_name} on {config.dataset.name}")
    logger.info(f"{'='*80}\n")
    
    output_dir = config.get_output_dir()
    dataset_name = config.dataset.name
    min_conf = config.benchmark.min_conf
    
    # Step 1: Run inference
    logger.info("Step 1/4: Running inference...")
    predictions_dir = output_dir / "predictions"
    inference_result = run_inference(
        model_name=model_name,
        dataset_name=dataset_name,
        images_root=Path(config.dataset.images_root),
        annotations_json=Path(config.dataset.annotations_json) if config.dataset.annotations_json else None,
        output_dir=predictions_dir,
        max_images=config.benchmark.max_images,
    )
    
    if not inference_result["success"]:
        logger.error(f"Inference failed: {inference_result.get('error')}")
        return False
    
    predictions_file = Path(inference_result["predictions_file"])
    logger.info(f"✓ Inference complete\n")
    
    # Step 2: Calculate metrics
    logger.info("Step 2/4: Calculating metrics...")
    metrics_dir = output_dir / "metrics"
    ground_truth_file = predictions_dir / "ground_truth.json"
    
    metrics_result = calculate_metrics(
        predictions_file=predictions_file,
        ground_truth_file=ground_truth_file if ground_truth_file.exists() else None,
        output_dir=metrics_dir,
        min_conf=min_conf,
    )
    
    if not metrics_result["success"]:
        logger.error(f"Metrics calculation failed: {metrics_result.get('error')}")
        return False
    
    logger.info(f"✓ Metrics calculated\n")
    
    # Step 3: Calculate per-joint errors (if ground truth available)
    if ground_truth_file.exists():
        logger.info("Step 3/4: Calculating per-joint errors...")
        per_joint_result = calculate_per_joint_errors(
            predictions_file=predictions_file,
            ground_truth_file=ground_truth_file,
            output_dir=metrics_dir,
            min_conf=min_conf,
        )
        
        if per_joint_result["success"]:
            logger.info(f"✓ Per-joint errors calculated\n")
        else:
            logger.warning(f"Per-joint error calculation failed\n")
    else:
        logger.info("Step 3/4: Skipping per-joint errors (no ground truth)\n")
    
    # Step 4: Generate overlays
    logger.info("Step 4/4: Generating skeleton overlays...")
    overlay_dir = output_dir / "overlays" / model_name / dataset_name
    overlay_result = generate_overlays(
        predictions_file=predictions_file,
        images_root=Path(config.dataset.images_root),
        output_dir=overlay_dir,
        min_conf=min_conf,
    )
    
    if not overlay_result["success"]:
        logger.error(f"Overlay generation failed: {overlay_result.get('error')}")
        return False
    
    logger.info(f"✓ Overlays generated\n")
    
    logger.info(f"{'='*80}")
    logger.info(f"Benchmark complete for {model_name}!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}\n")
    
    return True


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Run complete benchmark for a single model on a dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to benchmark",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run single model benchmark
    success = run_single_benchmark(args.model, config)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
