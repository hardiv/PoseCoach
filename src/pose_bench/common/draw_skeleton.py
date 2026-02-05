"""Utilities for drawing pose skeletons on images."""

import cv2
import numpy as np
from pathlib import Path

from .coco_schema import COCO_SKELETON_EDGES


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    conf: np.ndarray,
    min_conf: float = 0.2,
    joint_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw pose skeleton on image.
    
    Args:
        image: BGR image (H, W, 3)
        keypoints: Joint coordinates (17, 2) in pixel space
        conf: Joint confidences (17,)
        min_conf: Minimum confidence threshold for visualization
        joint_radius: Radius of joint circles
        line_thickness: Thickness of skeleton lines
        
    Returns:
        Image with skeleton overlay (copy of input)
    """
    output = image.copy()
    
    # Draw skeleton edges (bones)
    for idx1, idx2 in COCO_SKELETON_EDGES:
        if conf[idx1] >= min_conf and conf[idx2] >= min_conf:
            pt1 = keypoints[idx1]
            pt2 = keypoints[idx2]
            
            # Check for valid coordinates (not NaN)
            if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                cv2.line(output, pt1, pt2, (0, 255, 0), line_thickness)
    
    # Draw joints (keypoints)
    for idx in range(len(keypoints)):
        if conf[idx] >= min_conf:
            pt = keypoints[idx]
            if not np.isnan(pt).any():
                pt = tuple(map(int, pt))
                # Color code: high conf = green, medium = yellow, low = red
                if conf[idx] > 0.7:
                    color = (0, 255, 0)  # Green
                elif conf[idx] > 0.4:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.circle(output, pt, joint_radius, color, -1)
                cv2.circle(output, pt, joint_radius + 1, (0, 0, 0), 1)  # Black outline
    
    return output


def save_overlay_image(
    image: np.ndarray,
    keypoints: np.ndarray,
    conf: np.ndarray,
    output_path: str | Path,
    min_conf: float = 0.2,
) -> None:
    """
    Draw skeleton and save overlay image.
    
    Args:
        image: BGR image
        keypoints: Joint coordinates (17, 2)
        conf: Joint confidences (17,)
        output_path: Path to save overlay image
        min_conf: Minimum confidence threshold
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    overlay = draw_skeleton(image, keypoints, conf, min_conf)
    cv2.imwrite(str(output_path), overlay)
