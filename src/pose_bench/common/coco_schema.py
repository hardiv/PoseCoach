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
