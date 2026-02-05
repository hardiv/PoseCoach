"""MediaPipe BlazePose implementation."""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import urllib.request
import os

from .base import PoseEstimator, PoseResult
from ..common.coco_schema import get_mediapipe_to_coco_mapping, create_empty_coco17_result


# Default model URL
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
DEFAULT_MODEL_NAME = "pose_landmarker_heavy.task"


def download_model(model_dir: Path = Path.home() / ".pose_bench" / "models") -> str:
    """
    Download MediaPipe pose landmarker model if not already present.
    
    Args:
        model_dir: Directory to store the model
        
    Returns:
        Path to the model file
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / DEFAULT_MODEL_NAME
    
    if not model_path.exists():
        print(f"Downloading MediaPipe model to {model_path}...")
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
        print("Download complete!")
    
    return str(model_path)


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe BlazePose model adapter for COCO-17 format."""
    
    name = "mediapipe"
    
    def __init__(
        self,
        model_asset_path: str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Pose using the Tasks API.
        
        Args:
            model_asset_path: Path to pose landmarker model file (.task)
                             If None, downloads the default model automatically
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        # Download model if not provided
        if model_asset_path is None:
            model_asset_path = download_model()
        
        # Configure the pose landmarker
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.mp_to_coco = get_mediapipe_to_coco_mapping()
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Run MediaPipe pose estimation and convert to COCO-17 format.
        
        Args:
            bgr_image: Input image in BGR format (H, W, 3)
            
        Returns:
            PoseResult with COCO-17 keypoints
        """
        h, w = bgr_image.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = bgr_image[:, :, ::-1].copy()
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Run inference
        detection_result = self.landmarker.detect(mp_image)
        
        # Initialize empty COCO-17 arrays
        keypoints, conf = create_empty_coco17_result()
        
        # Use the first detected person (strongest detection)
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            # Map MediaPipe landmarks to COCO-17
            for coco_idx, mp_idx in self.mp_to_coco.items():
                lm = landmarks[mp_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = lm.x * w
                y = lm.y * h
                
                # Use visibility as confidence (clamp to [0, 1])
                confidence = np.clip(lm.visibility, 0.0, 1.0)
                
                keypoints[coco_idx] = [x, y]
                conf[coco_idx] = confidence
        
        # Compute rough bounding box from detected keypoints
        bbox = None
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_points) > 0:
            x_min, y_min = valid_points.min(axis=0)
            x_max, y_max = valid_points.max(axis=0)
            bbox = (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))
        
        return PoseResult(
            keypoints=keypoints,
            conf=conf,
            bbox=bbox,
            model_name=self.name,
        )
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
