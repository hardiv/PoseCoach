"""COCO keypoint schema and mapping utilities."""

import numpy as np
from typing import Dict

# COCO-17 keypoint names (standard order)
COCO_KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# COCO skeleton edges (pairs of keypoint indices)
COCO_SKELETON_EDGES = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (0, 5),   # nose -> left_shoulder
    (0, 6),   # nose -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]


def get_mediapipe_to_coco_mapping() -> Dict[int, int]:
    """
    Get mapping from MediaPipe Pose landmark indices to COCO-17 indices.
    
    MediaPipe has 33 landmarks. We map the relevant ones to COCO-17.
    Returns dict mapping COCO index -> MediaPipe landmark index.
    """
    # MediaPipe landmark indices (from mediapipe.solutions.pose.PoseLandmark)
    return {
        0: 0,    # nose
        1: 2,    # left_eye (left_eye_inner)
        2: 5,    # right_eye (right_eye_inner)
        3: 7,    # left_ear
        4: 8,    # right_ear
        5: 11,   # left_shoulder
        6: 12,   # right_shoulder
        7: 13,   # left_elbow
        8: 14,   # right_elbow
        9: 15,   # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }


def create_empty_coco17_result() -> tuple[np.ndarray, np.ndarray]:
    """
    Create empty COCO-17 keypoints and confidence arrays.
    
    Returns:
        keypoints: (17, 2) array filled with NaN
        conf: (17,) array filled with 0.0
    """
    keypoints = np.full((17, 2), np.nan, dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    return keypoints, conf


# MPII 16-joint keypoint names (standard order)
MPII_KEYPOINT_NAMES = [
    "right_ankle",     # 0
    "right_knee",      # 1
    "right_hip",       # 2
    "left_hip",        # 3
    "left_knee",       # 4
    "left_ankle",      # 5
    "pelvis",          # 6
    "thorax",          # 7
    "upper_neck",      # 8
    "head_top",        # 9
    "right_wrist",     # 10
    "right_elbow",     # 11
    "right_shoulder",  # 12
    "left_shoulder",   # 13
    "left_elbow",      # 14
    "left_wrist",      # 15
]


def map_mpii_to_coco17(mpii_keypoints: np.ndarray, mpii_visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map MPII 16-joint keypoints to COCO-17 format.
    
    MPII lacks: nose, eyes, ears. We can approximate head/neck but leave face joints as NaN.
    
    Args:
        mpii_keypoints: (16, 2) MPII joint coordinates
        mpii_visible: (16,) MPII visibility flags (1=visible, 0=not visible)
        
    Returns:
        coco_keypoints: (17, 2) COCO-17 joint coordinates
        coco_conf: (17,) COCO-17 confidence (1.0 for visible, 0.0 for not visible)
    """
    coco_keypoints, coco_conf = create_empty_coco17_result()
    
    # Direct mappings where joints match
    # COCO index -> (MPII index, name)
    mapping = {
        # Face joints: no direct mapping from MPII, leave as NaN
        # 0: nose - no mapping
        # 1: left_eye - no mapping
        # 2: right_eye - no mapping
        # 3: left_ear - no mapping
        # 4: right_ear - no mapping
        
        # Upper body
        5: (13, "left_shoulder"),   # left_shoulder
        6: (12, "right_shoulder"),  # right_shoulder
        7: (14, "left_elbow"),      # left_elbow
        8: (11, "right_elbow"),     # right_elbow
        9: (15, "left_wrist"),      # left_wrist
        10: (10, "right_wrist"),    # right_wrist
        
        # Lower body
        11: (3, "left_hip"),        # left_hip
        12: (2, "right_hip"),       # right_hip
        13: (4, "left_knee"),       # left_knee
        14: (1, "right_knee"),      # right_knee
        15: (5, "left_ankle"),      # left_ankle
        16: (0, "right_ankle"),     # right_ankle
    }
    
    for coco_idx, (mpii_idx, _) in mapping.items():
        if mpii_visible[mpii_idx] > 0:
            coco_keypoints[coco_idx] = mpii_keypoints[mpii_idx]
            coco_conf[coco_idx] = 1.0
    
    return coco_keypoints, coco_conf
