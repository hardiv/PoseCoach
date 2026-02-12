"""
Generate skeleton overlay visualizations from predictions.

This script:
1. Loads predictions from JSON
2. Loads original images
3. Draws skeleton overlays on images
4. Saves overlay images to output directory
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .common.draw_skeleton import save_overlay_image


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def generate_overlays(
    predictions_file: Path,
    images_root: Path,
    output_dir: Path,
    min_conf: float = 0.3,
) -> Dict:
    """
    Generate skeleton overlay visualizations.
    
    Args:
        predictions_file: Path to predictions JSON
        images_root: Path to directory containing original images
        output_dir: Directory to save overlay images
        min_conf: Minimum confidence threshold for drawing joints
        
    Returns:
        Dictionary with generation statistics
    """
    logger = setup_logging()
    
    logger.info(f"Generating overlays from {predictions_file}")
    logger.info(f"Images root: {images_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Min confidence: {min_conf}")
    
    # Load predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    
    if not predictions:
        logger.error("No predictions found!")
        return {"success": False, "error": "No predictions"}
    
    model_name = predictions_file.stem.replace("_predictions", "")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each prediction
    successful = 0
    failed = []
    
    for pred in tqdm(predictions, desc="Generating overlays", unit="img"):
        try:
            image_name = pred["image_name"]
            keypoints = np.array(pred["keypoints"])  # (17, 2)
            confidences = np.array(pred["confidences"])  # (17,)
            
            # Find and load original image
            # Try direct path first
            image_path = Path(pred.get("image_path", images_root / image_name))
            
            if not image_path.exists():
                # Try searching in images_root
                image_path = images_root / image_name
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_name}")
                failed.append(image_name)
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_name}")
                failed.append(image_name)
                continue
            
            # Save overlay
            overlay_path = output_dir / image_name
            save_overlay_image(
                image,
                keypoints,
                confidences,
                overlay_path,
                min_conf,
            )
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing {pred.get('image_name', 'unknown')}: {e}")
            failed.append(pred.get("image_name", "unknown"))
            continue
    
    # Log summary
    logger.info(f"\nOverlay generation complete!")
    logger.info(f"  Successful: {successful}/{len(predictions)}")
    logger.info(f"  Failed: {len(failed)}")
    
    if failed:
        logger.warning(f"  Failed images: {', '.join(failed[:10])}" + 
                      (f" and {len(failed) - 10} more..." if len(failed) > 10 else ""))
    
    stats = {
        "model_name": model_name,
        "total_predictions": len(predictions),
        "successful_overlays": successful,
        "failed_overlays": len(failed),
    }
    
    # Save stats
    stats_file = output_dir / "overlay_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    return {
        "success": True,
        "stats": stats,
        "output_dir": str(output_dir),
    }


def main():
    """Main entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Generate skeleton overlay visualizations from predictions"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Path to directory containing original images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save overlay images",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for drawing joints",
    )
    
    args = parser.parse_args()
    
    result = generate_overlays(
        predictions_file=args.predictions,
        images_root=args.images_root,
        output_dir=args.output_dir,
        min_conf=args.min_conf,
    )
    
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
