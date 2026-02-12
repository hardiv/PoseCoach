"""
Run pose estimation inference on a dataset and save predictions.

This script:
1. Loads images from a dataset
2. Runs a pose estimation model on each image
3. Saves predictions (keypoints, confidences) to JSON
4. Saves ground truth annotations if available
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .common.io import load_coco_annotations, get_image_paths
from .common.coco_schema import map_mpii_to_coco17
from .datasets.mpii import get_mpii_records
from .models import get_model, MODEL_REGISTRY


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def run_inference(
    model_name: str,
    dataset_name: str,
    images_root: Path,
    annotations_json: Optional[Path],
    output_dir: Path,
    max_images: Optional[int] = None,
) -> Dict:
    """
    Run inference on a dataset with a specific model.
    
    Args:
        model_name: Name of the model to use
        dataset_name: Dataset identifier (coco, mpii, gym_exercises)
        images_root: Path to images directory
        annotations_json: Path to annotations (optional for gym exercises)
        output_dir: Directory to save predictions
        max_images: Maximum number of images to process
        
    Returns:
        Dictionary with inference statistics
    """
    logger = setup_logging(output_dir / "inference.log")
    
    logger.info(f"Running inference: {model_name} on {dataset_name}")
    logger.info(f"Images root: {images_root}")
    logger.info(f"Output directory: {output_dir}")
    
    # Determine dataset type
    if "coco" in dataset_name.lower():
        dataset_type = "coco"
    elif "mpii" in dataset_name.lower():
        dataset_type = "mpii"
    else:
        dataset_type = "gym"
    
    # Load dataset
    image_paths = []
    dataset_records = None
    
    if dataset_type == "coco":
        logger.info(f"Loading COCO annotations from {annotations_json}")
        annotations = load_coco_annotations(str(annotations_json))
        image_paths = get_image_paths(images_root, annotations, max_images)
        
    elif dataset_type == "mpii":
        logger.info(f"Loading MPII annotations from {annotations_json}")
        dataset_records = get_mpii_records(
            str(images_root),
            str(annotations_json),
            max_images,
        )
        image_paths = [Path(rec["image_path"]) for rec in dataset_records]
        
    else:  # gym exercises - no annotations
        logger.info(f"Loading gym exercise images from {images_root}")
        image_paths = list(images_root.glob("*.jpg")) + list(images_root.glob("*.png"))
        if max_images:
            image_paths = image_paths[:max_images]
    
    if not image_paths:
        logger.error("No images found!")
        return {"success": False, "error": "No images found"}
    
    logger.info(f"Processing {len(image_paths)} images")
    
    # Initialize model
    try:
        model = get_model(model_name)
        logger.info(f"Initialized model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return {"success": False, "error": str(e)}
    
    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    ground_truths = []
    inference_times = []
    
    # Process each image
    failed_images = []
    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Inference ({model_name})", unit="img")):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                failed_images.append(str(img_path))
                continue
            
            # Run inference with timing
            start_time = time.perf_counter()
            result = model.predict(image)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000.0
            
            # Store prediction
            prediction = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "keypoints": result["keypoints"].tolist(),  # (17, 2)
                "confidences": result["conf"].tolist(),  # (17,)
                "inference_time_ms": inference_time_ms,
            }
            predictions.append(prediction)
            inference_times.append(inference_time_ms)
            
            # Store ground truth if available (MPII)
            if dataset_type == "mpii" and dataset_records:
                record = dataset_records[idx]
                gt_keypoints, gt_visible = map_mpii_to_coco17(
                    record["keypoints"],
                    record["visible"]
                )
                ground_truth = {
                    "image_name": img_path.name,
                    "keypoints": gt_keypoints.tolist(),
                    "visible": gt_visible.tolist(),
                }
                ground_truths.append(ground_truth)
                
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            failed_images.append(str(img_path))
            continue
    
    # Save predictions
    predictions_file = output_dir / f"{model_name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Save ground truth if available
    if ground_truths:
        gt_file = output_dir / "ground_truth.json"
        with open(gt_file, "w") as f:
            json.dump(ground_truths, f, indent=2)
        logger.info(f"Saved ground truth to {gt_file}")
    
    # Save statistics
    stats = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_images": len(image_paths),
        "successful_predictions": len(predictions),
        "failed_images": len(failed_images),
        "mean_inference_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "std_inference_time_ms": float(np.std(inference_times)) if inference_times else 0.0,
        "has_ground_truth": len(ground_truths) > 0,
    }
    
    stats_file = output_dir / f"{model_name}_inference_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Log summary
    logger.info(f"\nInference complete!")
    logger.info(f"  Successful: {len(predictions)}/{len(image_paths)}")
    logger.info(f"  Failed: {len(failed_images)}")
    logger.info(f"  Mean inference time: {stats['mean_inference_time_ms']:.2f} ms")
    
    return {
        "success": True,
        "predictions_file": str(predictions_file),
        "stats": stats,
    }


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Run pose estimation inference on a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for inference",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset identifier (e.g., coco, mpii, gym_exercises)",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Path to images directory",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Path to annotations JSON (optional for gym exercises)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (null = all)",
    )
    
    args = parser.parse_args()
    
    result = run_inference(
        model_name=args.model,
        dataset_name=args.dataset_name,
        images_root=args.images_root,
        annotations_json=args.annotations_json,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
    
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
