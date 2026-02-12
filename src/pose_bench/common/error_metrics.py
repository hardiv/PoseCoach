"""
Standardized pose estimation error metrics.

Implements:
- PCK (Percentage of Correct Keypoints) - MPII standard
- OKS (Object Keypoint Similarity) - COCO standard  
- Normalized pixel error - scale-invariant comparison
"""

import numpy as np
from typing import Tuple, Optional


# COCO OKS sigmas - per-joint fall-off constants
# These define how sensitive each joint is to error
# Smaller sigma = less tolerance for error (e.g., eyes)
# Larger sigma = more tolerance (e.g., hips)
COCO_OKS_SIGMAS = np.array([
    0.026,  # 0: nose
    0.025,  # 1: left_eye
    0.025,  # 2: right_eye
    0.035,  # 3: left_ear
    0.035,  # 4: right_ear
    0.079,  # 5: left_shoulder
    0.079,  # 6: right_shoulder
    0.072,  # 7: left_elbow
    0.072,  # 8: right_elbow
    0.062,  # 9: left_wrist
    0.062,  # 10: right_wrist
    0.107,  # 11: left_hip
    0.107,  # 12: right_hip
    0.087,  # 13: left_knee
    0.087,  # 14: right_knee
    0.089,  # 15: left_ankle
    0.089,  # 16: right_ankle
])


def calculate_pck(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_visible: np.ndarray,
    threshold: float = 0.5,
    normalize_method: str = "head",
) -> Tuple[float, np.ndarray]:
    """
    Calculate PCK (Percentage of Correct Keypoints).
    
    A keypoint is "correct" if within threshold * reference_distance.
    Standard MPII uses PCK@0.5 with head segment normalization.
    
    Args:
        pred_keypoints: Predicted keypoints (17, 2)
        gt_keypoints: Ground truth keypoints (17, 2)
        gt_visible: Visibility flags (17,)
        threshold: Distance threshold as fraction of reference (0.5 = 50%)
        normalize_method: "head" (head segment), "torso" (torso diagonal), or "bbox" (bbox diagonal)
        
    Returns:
        pck_score: Overall PCK (0-1)
        per_joint_pck: Per-joint PCK scores (17,)
    """
    # Calculate reference distance based on method
    if normalize_method == "head":
        # Head segment: distance between top of head (nose) and neck (midpoint of shoulders)
        nose = gt_keypoints[0]  # nose
        l_shoulder = gt_keypoints[5]
        r_shoulder = gt_keypoints[6]
        neck = (l_shoulder + r_shoulder) / 2
        ref_dist = np.linalg.norm(nose - neck)
        
    elif normalize_method == "torso":
        # Torso diagonal: distance from left shoulder to right hip
        l_shoulder = gt_keypoints[5]
        r_hip = gt_keypoints[12]
        ref_dist = np.linalg.norm(l_shoulder - r_hip)
        
    elif normalize_method == "bbox":
        # Bounding box diagonal
        valid_gt = gt_keypoints[gt_visible == 1]
        if len(valid_gt) > 0:
            bbox_min = valid_gt.min(axis=0)
            bbox_max = valid_gt.max(axis=0)
            ref_dist = np.linalg.norm(bbox_max - bbox_min)
        else:
            ref_dist = 1.0  # Fallback
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")
    
    # Avoid division by zero
    if ref_dist == 0 or np.isnan(ref_dist):
        ref_dist = 1.0
    
    # Calculate per-joint distances
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    
    # Normalize by reference distance
    normalized_distances = distances / ref_dist
    
    # Check which joints are correct (within threshold)
    correct = (normalized_distances <= threshold) & (gt_visible == 1)
    
    # Per-joint PCK
    per_joint_pck = np.zeros(17)
    for i in range(17):
        if gt_visible[i]:
            per_joint_pck[i] = float(correct[i])
        else:
            per_joint_pck[i] = np.nan  # Not visible, no score
    
    # Overall PCK (only over visible joints)
    num_visible = gt_visible.sum()
    if num_visible > 0:
        pck_score = correct.sum() / num_visible
    else:
        pck_score = np.nan
    
    return float(pck_score), per_joint_pck


def calculate_oks(
    pred_keypoints: np.ndarray,
    pred_conf: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_visible: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    sigmas: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Calculate OKS (Object Keypoint Similarity) - COCO standard metric.
    
    OKS = Σ exp(-d²/(2*s²*k²)) * δ(visible) / Σ δ(visible)
    where:
    - d = normalized distance
    - s = object scale (√area of bbox)
    - k = per-joint falloff constant (sigma)
    
    Args:
        pred_keypoints: Predicted keypoints (17, 2)
        pred_conf: Prediction confidences (17,)
        gt_keypoints: Ground truth keypoints (17, 2)
        gt_visible: Visibility flags (17,)
        bbox: Bounding box (x, y, w, h) for scale calculation
        sigmas: Per-joint falloff constants (17,), default: COCO_OKS_SIGMAS
        
    Returns:
        oks_score: Overall OKS (0-1, higher is better)
        per_joint_oks: Per-joint OKS contributions (17,)
    """
    if sigmas is None:
        sigmas = COCO_OKS_SIGMAS
    
    # Calculate object scale from bbox
    if bbox is not None:
        x, y, w, h = bbox
        scale = np.sqrt(w * h)
    else:
        # Fallback: use distance between keypoints
        valid_gt = gt_keypoints[gt_visible == 1]
        if len(valid_gt) > 0:
            bbox_min = valid_gt.min(axis=0)
            bbox_max = valid_gt.max(axis=0)
            dims = bbox_max - bbox_min
            scale = np.sqrt(dims[0] * dims[1])
        else:
            scale = 1.0
    
    # Avoid division by zero
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    
    # Calculate per-joint distances
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    
    # Calculate OKS per joint: exp(-d²/(2*s²*k²))
    per_joint_oks = np.zeros(17)
    for i in range(17):
        if gt_visible[i]:
            d = distances[i]
            k = sigmas[i]
            # OKS formula
            oks_value = np.exp(-(d ** 2) / (2 * (scale ** 2) * (k ** 2)))
            per_joint_oks[i] = oks_value
        else:
            per_joint_oks[i] = np.nan  # Not visible
    
    # Overall OKS (average over visible joints)
    num_visible = gt_visible.sum()
    if num_visible > 0:
        oks_score = np.nansum(per_joint_oks) / num_visible
    else:
        oks_score = np.nan
    
    return float(oks_score), per_joint_oks


def calculate_normalized_error(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_visible: np.ndarray,
    normalize_method: str = "bbox",
) -> Tuple[float, np.ndarray]:
    """
    Calculate normalized pixel error.
    
    Normalizes by a reference distance to make errors comparable
    across different image sizes and person scales.
    
    Args:
        pred_keypoints: Predicted keypoints (17, 2)
        gt_keypoints: Ground truth keypoints (17, 2)
        gt_visible: Visibility flags (17,)
        normalize_method: "bbox" (bbox diagonal), "torso" (torso diagonal), "height" (person height)
        
    Returns:
        mean_normalized_error: Mean normalized error across visible joints
        per_joint_normalized_error: Per-joint normalized errors (17,)
    """
    # Calculate reference distance
    valid_gt = gt_keypoints[gt_visible == 1]
    
    if len(valid_gt) == 0:
        return np.nan, np.full(17, np.nan)
    
    if normalize_method == "bbox":
        # Bounding box diagonal
        bbox_min = valid_gt.min(axis=0)
        bbox_max = valid_gt.max(axis=0)
        ref_dist = np.linalg.norm(bbox_max - bbox_min)
        
    elif normalize_method == "torso":
        # Torso diagonal
        l_shoulder = gt_keypoints[5]
        r_hip = gt_keypoints[12]
        ref_dist = np.linalg.norm(l_shoulder - r_hip)
        
    elif normalize_method == "height":
        # Person height (top to bottom of visible keypoints)
        bbox_min = valid_gt.min(axis=0)
        bbox_max = valid_gt.max(axis=0)
        ref_dist = bbox_max[1] - bbox_min[1]  # Height only
        
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")
    
    # Avoid division by zero
    if ref_dist == 0 or np.isnan(ref_dist):
        ref_dist = 1.0
    
    # Calculate per-joint distances
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    
    # Normalize
    per_joint_normalized_error = distances / ref_dist
    
    # Mask invisible joints
    per_joint_normalized_error = np.where(
        gt_visible == 1,
        per_joint_normalized_error,
        np.nan
    )
    
    # Mean over visible joints
    mean_normalized_error = float(np.nanmean(per_joint_normalized_error))
    
    return mean_normalized_error, per_joint_normalized_error
