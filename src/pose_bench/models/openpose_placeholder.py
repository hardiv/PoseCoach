"""OpenPose placeholder implementation."""

import numpy as np
from .base import PoseEstimator, PoseResult


class OpenPoseEstimator(PoseEstimator):
    """
    OpenPose model adapter (PLACEHOLDER).
    
    To implement:
    1. Install OpenPose using Docker or build from source:
       Docker: docker pull cwaffles/openpose
       
    2. Run OpenPose inference via Docker or Python API:
       - Docker: Mount image directory and run inference
       - Python: Use pyopenpose bindings if available
       
    3. Parse OpenPose output (JSON with BODY_25 or COCO format)
    
    4. Map keypoints to COCO-17 format:
       - If using BODY_25: map 25 keypoints -> COCO-17
       - If using COCO: direct mapping (18 -> 17, drop neck)
       
    5. Return PoseResult with keypoints, confidence, and bbox
    
    Example Docker command:
        docker run --rm -v /path/to/images:/images cwaffles/openpose \
            --image_dir /images --write_json /images/output --display 0
    
    TODO: Implement actual OpenPose integration
    """
    
    name = "openpose"
    
    def __init__(self):
        raise NotImplementedError(
            "OpenPose adapter not yet implemented. See docstring for integration guide."
        )
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        raise NotImplementedError("OpenPose not implemented")
