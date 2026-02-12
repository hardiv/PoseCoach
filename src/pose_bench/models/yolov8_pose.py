"""YOLOv8-Pose implementation using Ultralytics."""

import numpy as np
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .base import PoseEstimator, PoseResult
from ..common.coco_schema import create_empty_coco17_result


class YOLOv8PoseEstimator(PoseEstimator):
    """
    YOLOv8-Pose model adapter.
    
    Uses Ultralytics YOLOv8 pose estimation models.
    Automatically downloads model weights on first use.
    """
    
    name = "yolov8-pose"
    
    def __init__(
        self,
        model_size: str = "m",  # n, s, m, l, x
        device: str = "cpu",
    ):
        """
        Initialize YOLOv8-Pose.
        
        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            device: Device to run on ('cpu' or 'cuda')
        """
        model_name = f"yolov8{model_size}-pose.pt"
        print(f"Loading YOLOv8-Pose model: {model_name}")
        
        # YOLO will automatically download the model if not present
        self.model = YOLO(model_name)
        self.model.to(device)
        
        print(f"Initialized {model_name} on {device}")
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Run YOLOv8-Pose on a single image.
        
        Args:
            bgr_image: Input image in BGR format (H, W, 3)
            
        Returns:
            PoseResult with COCO-17 keypoints
        """
        # Run inference (verbose=False to suppress output)
        results = self.model(bgr_image, verbose=False)
        
        # Parse results
        if len(results) == 0 or results[0].keypoints is None:
            return self._empty_result()
        
        result = results[0]
        keypoints_data = result.keypoints
        
        # Check if any people detected
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return self._empty_result()
        
        # Take the first person (highest confidence)
        # YOLOv8 outputs: keypoints.xy shape (num_people, 17, 2)
        # YOLOv8 outputs: keypoints.conf shape (num_people, 17)
        xy = keypoints_data.xy[0].cpu().numpy()  # (17, 2)
        conf = keypoints_data.conf[0].cpu().numpy()  # (17,)
        
        # YOLOv8 uses COCO-17 format directly
        keypoints = np.full((17, 2), np.nan, dtype=np.float32)
        confidences = np.zeros(17, dtype=np.float32)
        
        for i in range(17):
            if conf[i] > 0:  # Valid keypoint
                keypoints[i] = xy[i]
                confidences[i] = conf[i]
        
        # Compute bounding box
        bbox = None
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        
        return PoseResult(
            keypoints=keypoints,
            conf=confidences,
            bbox=bbox,
            model_name=self.name,
        )
    
    def _empty_result(self) -> PoseResult:
        """Return empty result when no person detected."""
        keypoints, conf = create_empty_coco17_result()
        return PoseResult(
            keypoints=keypoints,
            conf=conf,
            bbox=None,
            model_name=self.name,
        )
