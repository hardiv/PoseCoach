"""MoveNet implementation using TensorFlow Hub."""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .base import PoseEstimator, PoseResult
from ..common.coco_schema import create_empty_coco17_result


class MoveNetEstimator(PoseEstimator):
    """
    MoveNet model adapter (Google).
    
    Uses TensorFlow Hub for MoveNet Lightning or Thunder models.
    Automatically downloads model on first use.
    """
    
    name = "movenet"
    
    def __init__(
        self,
        model_type: str = "lightning",  # "lightning" or "thunder"
    ):
        """
        Initialize MoveNet.
        
        Args:
            model_type: "lightning" (fast) or "thunder" (accurate)
        """
        if model_type not in ["lightning", "thunder"]:
            raise ValueError(f"model_type must be 'lightning' or 'thunder', got {model_type}")
        
        self.model_type = model_type
        
        # Model URLs from TensorFlow Hub
        model_urls = {
            "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
            "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        }
        
        model_url = model_urls[model_type]
        print(f"Loading MoveNet {model_type} from TF Hub...")
        
        # Load model from TensorFlow Hub
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        
        print(f"Initialized MoveNet {model_type}")
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Run MoveNet on a single image.
        
        Args:
            bgr_image: Input image in BGR format (H, W, 3)
            
        Returns:
            PoseResult with COCO-17 keypoints
        """
        h, w = bgr_image.shape[:2]
        
        # Convert BGR to RGB
        rgb_image = bgr_image[:, :, ::-1].copy()
        
        # Resize to model input size (192x192 for lightning, 256x256 for thunder)
        input_size = 192 if self.model_type == "lightning" else 256
        resized = tf.image.resize_with_pad(
            tf.convert_to_tensor(rgb_image),
            input_size,
            input_size
        )
        
        # Add batch dimension and convert to int32
        input_image = tf.cast(resized, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)
        
        # Run inference
        outputs = self.movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]
        
        # MoveNet output shape: (17, 3) where each row is [y, x, confidence]
        # Note: coordinates are normalized [0, 1]
        keypoints = np.full((17, 2), np.nan, dtype=np.float32)
        confidences = np.zeros(17, dtype=np.float32)
        
        for i in range(17):
            y_norm, x_norm, conf = keypoints_with_scores[i]
            
            if conf > 0:  # Valid keypoint
                # Convert normalized coordinates to pixel coordinates
                keypoints[i] = [x_norm * w, y_norm * h]
                confidences[i] = conf
        
        # Compute bounding box from detected keypoints
        bbox = None
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_points) > 0:
            x_min, y_min = valid_points.min(axis=0)
            x_max, y_max = valid_points.max(axis=0)
            bbox = (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))
        
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
