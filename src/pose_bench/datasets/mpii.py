"""MPII Human Pose dataset loader."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, TypedDict


class MPIIRecord(TypedDict):
    """Single MPII annotation record."""
    image_path: Path
    keypoints: np.ndarray  # (16, 2) joint coordinates
    visible: np.ndarray    # (16,) visibility flags (1=visible, 0=not visible)


def load_mpii_annotations(json_path: str | Path) -> Dict[str, Any]:
    """
    Load MPII annotations from JSON file.
    
    MPII annotations should be in JSON format with structure:
    {
        "images": [...],
        "annotations": [
            {
                "image_id": int,
                "joints": [[x, y], ...],  # 16 joints
                "joints_vis": [0/1, ...]  # 16 visibility flags
            },
            ...
        ]
    }
    
    Args:
        json_path: Path to MPII annotations JSON file
        
    Returns:
        Parsed annotations dict
    """
    with open(json_path, "r") as f:
        return json.load(f)


def get_mpii_records(
    images_root: str | Path,
    annotations_path: str | Path,
    max_images: int | None = None,
) -> List[MPIIRecord]:
    """
    Load MPII dataset records.
    
    Args:
        images_root: Directory containing MPII images
        annotations_path: Path to MPII annotations JSON
        max_images: Maximum number of images to load (None for all)
        
    Returns:
        List of MPII records with image paths, keypoints, and visibility
    """
    images_root = Path(images_root)
    annotations = load_mpii_annotations(annotations_path)
    
    # Build image id to filename mapping
    image_id_to_filename = {}
    for img in annotations.get("images", []):
        image_id_to_filename[img["id"]] = img["file_name"]
    
    records = []
    annots = annotations.get("annotations", [])
    
    if max_images is not None:
        annots = annots[:max_images]
    
    for ann in annots:
        image_id = ann["image_id"]
        if image_id not in image_id_to_filename:
            continue
        
        image_path = images_root / image_id_to_filename[image_id]
        
        # Skip if image doesn't exist
        if not image_path.exists():
            continue
        
        # Extract joints (16, 2)
        joints = np.array(ann["joints"], dtype=np.float32)
        
        # Extract visibility (16,)
        joints_vis = np.array(ann["joints_vis"], dtype=np.float32)
        
        records.append(MPIIRecord(
            image_path=image_path,
            keypoints=joints,
            visible=joints_vis,
        ))
    
    return records
