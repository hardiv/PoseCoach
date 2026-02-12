"""Pose estimation model implementations."""

from .base import PoseEstimator, PoseResult
from .mediapipe_blazepose import MediaPipePoseEstimator
from .yolov8_pose import YOLOv8PoseEstimator
from .movenet import MoveNetEstimator

# Model registry
MODEL_REGISTRY = {
    "mediapipe": MediaPipePoseEstimator,
    "yolov8-pose": YOLOv8PoseEstimator,
    "movenet": MoveNetEstimator,
    # Placeholders - not yet implemented
    # "openpose": OpenPoseEstimator,
    # "mmpose": MMPoseEstimator,
}


def get_model(model_name: str) -> PoseEstimator:
    """
    Get pose estimator instance by name.
    
    Args:
        model_name: Name of the model (e.g., 'mediapipe')
        
    Returns:
        Instantiated PoseEstimator
        
    Raises:
        ValueError: If model name is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class()


__all__ = ["PoseEstimator", "PoseResult", "MODEL_REGISTRY", "get_model"]
