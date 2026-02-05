"""I/O utilities for loading COCO data."""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_coco_annotations(json_path: str | Path) -> Dict[str, Any]:
    """
    Load COCO annotations JSON file.
    
    Args:
        json_path: Path to person_keypoints_val2017.json
        
    Returns:
        Parsed JSON data as dict
    """
    with open(json_path, "r") as f:
        return json.load(f)


def get_image_paths(
    images_root: str | Path,
    annotations: Dict[str, Any],
    max_images: int | None = None,
) -> List[Path]:
    """
    Get list of image paths from COCO annotations.
    
    Args:
        images_root: Directory containing val2017 images
        annotations: Parsed COCO annotations dict
        max_images: Maximum number of images to return (None for all)
        
    Returns:
        List of image file paths
    """
    images_root = Path(images_root)
    
    # Extract image filenames from annotations
    image_list = annotations["images"]
    
    if max_images is not None:
        image_list = image_list[:max_images]
    
    paths = [images_root / img["file_name"] for img in image_list]
    
    # Filter to only existing files
    existing_paths = [p for p in paths if p.exists()]
    
    if len(existing_paths) < len(paths):
        missing = len(paths) - len(existing_paths)
        print(f"Warning: {missing} image files not found, using {len(existing_paths)} images")
    
    return existing_paths
