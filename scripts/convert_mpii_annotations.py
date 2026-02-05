"""Convert MPII .mat annotations to JSON format for pose_bench."""

import scipy.io as sio
import json
import numpy as np
from pathlib import Path


def convert_mpii_to_json(mat_path, output_path, max_samples=None):
    """
    Convert MPII .mat annotations to JSON format.
    
    MPII 16-joint order:
    0: right_ankle, 1: right_knee, 2: right_hip, 3: left_hip,
    4: left_knee, 5: left_ankle, 6: pelvis, 7: thorax,
    8: upper_neck, 9: head_top, 10: right_wrist, 11: right_elbow,
    12: right_shoulder, 13: left_shoulder, 14: left_elbow, 15: left_wrist
    """
    print(f"Loading MPII annotations from {mat_path}...")
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    
    # Extract RELEASE structure
    release = mat['RELEASE']
    annolist = release.annolist
    
    # Build image list
    images = []
    annotations = []
    
    # Process all annotations
    count = 0
    for idx in range(len(annolist)):
        if max_samples and count >= max_samples:
            break
        
        anno = annolist[idx]
        
        # Get image info
        if not hasattr(anno, 'image'):
            continue
        img_name = anno.image.name if hasattr(anno.image, 'name') else str(anno.image)
        
        # Check if annotation has pose data
        if not hasattr(anno, 'annorect'):
            continue
        
        rects = anno.annorect
        if not isinstance(rects, np.ndarray):
            rects = [rects]
        
        # Process each person in the image (usually just one for single-person)
        for rect in rects:
            if not hasattr(rect, 'annopoints'):
                continue
            
            annopoints = rect.annopoints
            if not hasattr(annopoints, 'point'):
                continue
            
            points = annopoints.point
            
            # Initialize joints array (16, 2) and visibility (16,)
            joints = np.zeros((16, 2), dtype=np.float32)
            joints_vis = np.zeros(16, dtype=np.float32)
            
            # Fill in the joints
            if isinstance(points, np.ndarray):
                for point in points:
                    try:
                        joint_id = int(point.id)
                        if 0 <= joint_id < 16:
                            joints[joint_id] = [float(point.x), float(point.y)]
                            # Check if visibility is specified
                            if hasattr(point, 'is_visible'):
                                vis = point.is_visible
                                # Handle different numpy array types
                                if isinstance(vis, np.ndarray):
                                    joints_vis[joint_id] = float(vis.flat[0]) if vis.size > 0 else 1.0
                                else:
                                    joints_vis[joint_id] = float(vis)
                            else:
                                joints_vis[joint_id] = 1.0  # Assume visible if not specified
                    except (ValueError, AttributeError, IndexError):
                        continue
            else:
                # Single point
                try:
                    joint_id = int(points.id)
                    if 0 <= joint_id < 16:
                        joints[joint_id] = [float(points.x), float(points.y)]
                        if hasattr(points, 'is_visible'):
                            vis = points.is_visible
                            if isinstance(vis, np.ndarray):
                                joints_vis[joint_id] = float(vis.flat[0]) if vis.size > 0 else 1.0
                            else:
                                joints_vis[joint_id] = float(vis)
                        else:
                            joints_vis[joint_id] = 1.0
                except (ValueError, AttributeError, IndexError):
                    pass
            
            # Only include if we have at least 8 visible joints
            if joints_vis.sum() >= 8:
                # Add image if not already added
                if not any(img['file_name'] == img_name for img in images):
                    images.append({
                        'id': len(images),
                        'file_name': img_name,
                    })
                
                # Find image id
                img_id = next(img['id'] for img in images if img['file_name'] == img_name)
                
                annotations.append({
                    'image_id': img_id,
                    'joints': joints.tolist(),
                    'joints_vis': joints_vis.tolist(),
                })
                
                count += 1
    
    # Create final JSON structure
    data = {
        'images': images,
        'annotations': annotations,
    }
    
    print(f"Converted {len(annotations)} annotations from {len(images)} images")
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MPII .mat to JSON")
    parser.add_argument(
        "--mat-path",
        type=str,
        default="data/mpii/annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat",
        help="Path to MPII .mat file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mpii/annotations/mpii_annotations.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (for testing)"
    )
    
    args = parser.parse_args()
    
    convert_mpii_to_json(args.mat_path, args.output, args.max_samples)
