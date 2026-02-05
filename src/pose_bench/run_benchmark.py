"""Main benchmarking runner script."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .common.io import load_coco_annotations, get_image_paths
from .common.draw_skeleton import save_overlay_image
from .common.metrics import PoseMetrics
from .common.coco_schema import map_mpii_to_coco17
from .datasets.mpii import get_mpii_records
from .models import get_model, MODEL_REGISTRY


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "errors.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_benchmark(config: Config) -> None:
    """
    Run pose estimation benchmark.
    
    Args:
        config: Benchmark configuration
    """
    output_dir = config.get_output_dir()
    setup_logging(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting pose estimation benchmark")
    logger.info(f"Configuration: {config}")
    
    dataset_type = config.get_dataset_type()
    logger.info(f"Dataset type: {dataset_type}")
    
    # Load dataset based on type
    if dataset_type == "coco":
        # Load COCO annotations
        logger.info(f"Loading COCO annotations from {config.dataset.annotations_json}")
        try:
            annotations = load_coco_annotations(config.dataset.annotations_json)
        except FileNotFoundError:
            logger.error(f"Annotations file not found: {config.dataset.annotations_json}")
            logger.error("Please run: bash scripts/download_coco.sh")
            sys.exit(1)
        
        # Get image paths
        image_paths = get_image_paths(
            config.dataset.images_root,
            annotations,
            config.benchmark.max_images,
        )
        
        if not image_paths:
            logger.error("No images found. Please download COCO dataset.")
            sys.exit(1)
        
        logger.info(f"Processing {len(image_paths)} COCO images")
        dataset_records = None  # COCO doesn't need records format
        
    elif dataset_type == "mpii":
        # Load MPII dataset
        logger.info(f"Loading MPII annotations from {config.dataset.annotations_json}")
        try:
            dataset_records = get_mpii_records(
                config.dataset.images_root,
                config.dataset.annotations_json,
                config.benchmark.max_images,
            )
        except FileNotFoundError:
            logger.error(f"Annotations file not found: {config.dataset.annotations_json}")
            logger.error("Please run: bash scripts/download_mpii.sh")
            sys.exit(1)
        
        if not dataset_records:
            logger.error("No MPII records found. Please download MPII dataset.")
            sys.exit(1)
        
        logger.info(f"Processing {len(dataset_records)} MPII images")
        image_paths = [rec["image_path"] for rec in dataset_records]
        
    else:
        logger.error(f"Unknown dataset type: {dataset_type}. Supported: coco, mpii")
        sys.exit(1)
    
    # Initialize metrics tracker
    metrics = PoseMetrics(min_conf=config.benchmark.min_conf)
    
    # Run each model
    for model_name in config.benchmark.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running model: {model_name}")
        logger.info(f"{'='*60}")
        
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Model '{model_name}' not in registry. Skipping.")
            continue
        
        # Initialize model
        try:
            model = get_model(model_name)
            logger.info(f"Initialized {model_name}")
        except NotImplementedError as e:
            logger.warning(f"Model '{model_name}' not implemented: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to initialize '{model_name}': {e}")
            continue
        
        # Create output directories for this model and dataset
        overlay_dir = output_dir / "overlays" / model_name / config.dataset.name
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for idx, img_path in enumerate(tqdm(image_paths, desc=f"{model_name}", unit="img")):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.error(f"Failed to load image: {img_path}")
                    continue
                
                # Run inference
                result = model.predict(image)
                
                # Save overlay
                overlay_path = overlay_dir / img_path.name
                save_overlay_image(
                    image,
                    result["keypoints"],
                    result["conf"],
                    overlay_path,
                    config.benchmark.min_conf,
                )
                
                # Get ground truth if MPII dataset
                gt_keypoints = None
                gt_visible = None
                if dataset_type == "mpii" and dataset_records:
                    record = dataset_records[idx]
                    # Map MPII ground truth to COCO-17 format
                    gt_keypoints, gt_visible = map_mpii_to_coco17(
                        record["keypoints"],
                        record["visible"]
                    )
                
                # Record metrics
                metrics.add_result(
                    image_name=img_path.name,
                    model_name=model_name,
                    keypoints=result["keypoints"],
                    conf=result["conf"],
                    gt_keypoints=gt_keypoints,
                    gt_visible=gt_visible,
                )
                
            except Exception as e:
                logger.error(f"Error processing {img_path.name} with {model_name}: {e}")
                continue
        
        # Save per-model metrics
        metrics.save_per_image_metrics(output_dir, model_name, config.dataset.name)
        logger.info(f"Completed {model_name}")
    
    # Compute and save leaderboard
    logger.info(f"\n{'='*60}")
    logger.info("Computing leaderboard")
    logger.info(f"{'='*60}")
    metrics.compute_leaderboard(output_dir)
    
    logger.info(f"\nBenchmark complete! Results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pose estimation benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Override max_images from config",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Override max_images if specified
    if args.max_images is not None:
        config.benchmark.max_images = args.max_images
    
    # Run benchmark
    run_benchmark(config)


if __name__ == "__main__":
    main()
