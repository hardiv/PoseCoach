"""MMPose placeholder implementation."""

import numpy as np
from .base import PoseEstimator, PoseResult


class MMPoseEstimator(PoseEstimator):
    """
    MMPose model adapter (PLACEHOLDER).
    
    To implement:
    1. Install MMPose and dependencies:
       pip install mmcv-full mmpose
       
    2. Download a pretrained model config + checkpoint:
       - Browse: https://github.com/open-mmlab/mmpose/tree/master/configs
       - Example: HRNet-W48 on COCO
       - Download .pth checkpoint file
       
    3. Initialize MMPose inference API:
       from mmpose.apis import init_pose_model, inference_top_down_pose_model
       
       model = init_pose_model(config_file, checkpoint_file, device='cuda:0')
       
    4. Run inference:
       - MMPose expects person bounding boxes as input
       - For simplicity, use full image bbox or run person detector first
       - Get keypoints in COCO format (17 keypoints)
       
    5. Return PoseResult with keypoints, confidence, and bbox
    
    Example:
        results = inference_top_down_pose_model(
            model, image, person_bboxes, format='xyxy'
        )
        keypoints = results[0]['keypoints']  # (17, 3) with x, y, conf
    
    TODO: Implement actual MMPose integration
    """
    
    name = "mmpose"
    
    def __init__(self):
        raise NotImplementedError(
            "MMPose adapter not yet implemented. See docstring for integration guide."
        )
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        raise NotImplementedError("MMPose not implemented")
