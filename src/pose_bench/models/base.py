"""Base class for pose estimation models."""

from abc import ABC, abstractmethod
from typing import TypedDict, Optional
import numpy as np


class PoseResult(TypedDict):
    """Standardized pose estimation result."""
    keypoints: np.ndarray  # shape (17, 2) in pixel coordinates
    conf: np.ndarray       # shape (17,) confidence scores [0, 1]
    bbox: Optional[tuple[float, float, float, float]]  # (x, y, w, h) or None
    model_name: str


class PoseEstimator(ABC):
    """Abstract base class for pose estimation models."""
    
    name: str  # Model identifier
    
    @abstractmethod
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single image.
        
        For multi-person detection, returns the pose with highest confidence.
        For benchmarking, we focus on single-person pose estimation.
        
        Args:
            bgr_image: Input image in BGR format (H, W, 3)
            
        Returns:
            PoseResult with COCO-17 keypoints, confidences, optional bbox, and model name
        """
        pass
