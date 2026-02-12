"""OpenPose implementation using Docker."""

import numpy as np
import json
import subprocess
import tempfile
import cv2
from pathlib import Path
from typing import Optional

from .base import PoseEstimator, PoseResult
from ..common.coco_schema import create_empty_coco17_result


# OpenPose BODY_25 to COCO-17 mapping
# BODY_25 indices: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
BODY25_TO_COCO17 = {
    0: 0,   # Nose
    15: 1,  # LEye -> left_eye
    16: 2,  # REye -> right_eye
    17: 3,  # LEar -> left_ear
    18: 4,  # REar -> right_ear
    5: 5,   # LShoulder -> left_shoulder
    2: 6,   # RShoulder -> right_shoulder
    6: 7,   # LElbow -> left_elbow
    3: 8,   # RElbow -> right_elbow
    7: 9,   # LWrist -> left_wrist
    4: 10,  # RWrist -> right_wrist
    12: 11, # LHip -> left_hip
    9: 12,  # RHip -> right_hip
    13: 13, # LKnee -> left_knee
    10: 14, # RKnee -> right_knee
    14: 15, # LAnkle -> left_ankle
    11: 16, # RAnkle -> right_ankle
}


class OpenPoseEstimator(PoseEstimator):
    """
    OpenPose model adapter using Docker.
    
    Requires Docker to be installed.
    Uses cwaffles/openpose Docker image for inference.
    """
    
    name = "openpose"
    
    def __init__(
        self,
        docker_image: str = "cwaffles/openpose",
        use_gpu: bool = False,
    ):
        """
        Initialize OpenPose via Docker.
        
        Args:
            docker_image: Docker image name
            use_gpu: Whether to use GPU (requires nvidia-docker)
        """
        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Docker available: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Docker is not available. Please install Docker:\n"
                "  macOS: https://docs.docker.com/desktop/install/mac-install/\n"
                "  Linux: https://docs.docker.com/engine/install/"
            )
        
        self.docker_image = docker_image
        self.use_gpu = use_gpu
        
        # Pull Docker image if not present
        print(f"Checking for Docker image: {docker_image}")
        try:
            subprocess.run(
                ["docker", "image", "inspect", docker_image],
                capture_output=True,
                check=True
            )
            print(f"Docker image {docker_image} found")
        except subprocess.CalledProcessError:
            print(f"Pulling Docker image {docker_image} (this may take a while)...")
            subprocess.run(
                ["docker", "pull", docker_image],
                check=True
            )
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Run OpenPose via Docker on a single image.
        
        Args:
            bgr_image: Input image in BGR format (H, W, 3)
            
        Returns:
            PoseResult with COCO-17 keypoints
        """
        # Create temporary directory for input/output
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save input image
            input_path = tmpdir / "input.jpg"
            cv2.imwrite(str(input_path), bgr_image)
            
            # Run OpenPose via Docker
            docker_cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",  # Use Rosetta 2 for x86 emulation on ARM Macs
                "-v", f"{tmpdir}:/data",
                self.docker_image,
                "./build/examples/openpose/openpose.bin",
                "--image_dir", "/data",
                "--write_json", "/data/output",
                "--display", "0",
                "--render_pose", "0",
                "--model_pose", "BODY_25",
                "--num_gpu", "0",  # Force CPU mode (no CUDA on macOS)
                "--num_gpu_start", "0",
            ]
            
            # Run OpenPose
            try:
                subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    check=True,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                print("OpenPose timed out, returning empty result")
                return self._empty_result()
            except subprocess.CalledProcessError as e:
                print(f"OpenPose failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
                return self._empty_result()
            
            # Read output JSON
            output_dir = tmpdir / "output"
            if not output_dir.exists():
                return self._empty_result()
            
            json_files = list(output_dir.glob("*.json"))
            if not json_files:
                return self._empty_result()
            
            with open(json_files[0], 'r') as f:
                result = json.load(f)
            
            # Parse OpenPose output
            return self._parse_openpose_output(result, bgr_image.shape)
    
    def _parse_openpose_output(self, result: dict, image_shape: tuple) -> PoseResult:
        """Parse OpenPose JSON output to COCO-17 format."""
        h, w = image_shape[:2]
        keypoints, conf = create_empty_coco17_result()
        
        people = result.get('people', [])
        if not people:
            return PoseResult(
                keypoints=keypoints,
                conf=conf,
                bbox=None,
                model_name=self.name,
            )
        
        # Take the first person (highest confidence)
        person = people[0]
        pose_keypoints = person.get('pose_keypoints_2d', [])
        
        # OpenPose format: [x1, y1, conf1, x2, y2, conf2, ...]
        # BODY_25 has 25 keypoints
        if len(pose_keypoints) < 75:  # 25 * 3
            return self._empty_result()
        
        # Map BODY_25 to COCO-17
        for coco_idx, body25_idx in BODY25_TO_COCO17.items():
            base_idx = body25_idx * 3
            x = pose_keypoints[base_idx]
            y = pose_keypoints[base_idx + 1]
            c = pose_keypoints[base_idx + 2]
            
            if c > 0:  # Valid keypoint
                keypoints[coco_idx] = [x, y]
                conf[coco_idx] = c
        
        # Compute bounding box
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
    
    def _empty_result(self) -> PoseResult:
        """Return empty result when OpenPose fails or finds no person."""
        keypoints, conf = create_empty_coco17_result()
        return PoseResult(
            keypoints=keypoints,
            conf=conf,
            bbox=None,
            model_name=self.name,
        )
