# Pose Bench

A lightweight Python benchmarking suite for 2D pose estimation models on COCO images.

## Features

- 🎯 **Model Agnostic**: Extensible interface for adding pose estimators (strategy pattern)
- 📊 **Fast Metrics**: Compute joint detection rates, confidence scores, and pose validity
- 🎨 **Visual Overlays**: Generate skeleton overlay images for qualitative assessment
- 🔧 **Simple Config**: YAML-based configuration for datasets and models
- 📦 **Minimal Setup**: Uses `uv` for fast dependency management

## Current Models

- ✅ **MediaPipe BlazePose** - Fully implemented
- 🚧 **OpenPose** - Placeholder (see integration guide in code)
- 🚧 **MMPose** - Placeholder (see integration guide in code)

## Project Structure

```
pose_bench/
├── pyproject.toml           # Dependencies and project metadata
├── config.yaml              # Benchmark configuration
├── src/pose_bench/
│   ├── run_benchmark.py     # Main runner script
│   ├── config.py            # Configuration management
│   ├── common/              # Shared utilities
│   │   ├── coco_schema.py   # COCO-17 keypoint definitions
│   │   ├── io.py            # Dataset loading
│   │   ├── draw_skeleton.py # Skeleton visualization
│   │   └── metrics.py       # Metrics computation
│   └── models/              # Pose estimator implementations
│       ├── base.py          # Abstract base class
│       ├── mediapipe_blazepose.py
│       ├── openpose_placeholder.py
│       └── mm_pose_placeholder.py
├── scripts/
│   └── download_coco.sh     # COCO dataset download script
└── outputs/                 # Generated results (gitignored)
    ├── overlays/            # Skeleton overlay images
    ├── metrics/             # Per-image CSV files
    └── leaderboard.csv      # Aggregate results
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment with uv
uv venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### 2. Download COCO Dataset

```bash
# Downloads COCO val2017 images + annotations (~1.2GB)
bash scripts/download_coco.sh
```

This creates:
- `data/coco/val2017/` - 5000 validation images
- `data/coco/annotations/person_keypoints_val2017.json`

### 3. Run Benchmark

```bash
# Run with default config (100 images, MediaPipe only)
python -m pose_bench.run_benchmark --config config.yaml

# Process more images
python -m pose_bench.run_benchmark --config config.yaml --max-images 500

# Process all images (5000)
python -m pose_bench.run_benchmark --config config.yaml --max-images 0
```

### 4. View Results

Results are saved to `outputs/`:

- **Overlays**: `outputs/overlays/{model}/{image}.jpg` - Visual skeletons
- **Metrics**: `outputs/metrics/{model}.csv` - Per-image metrics
- **Leaderboard**: `outputs/leaderboard.csv` - Aggregate performance

Example leaderboard:
```
model_name   num_images  pose_rate  mean_detected_joints  mean_conf
mediapipe           100      87.00                 14.23      0.745
```

## Configuration

Edit `config.yaml` to customize:

```yaml
dataset:
  images_root: "data/coco/val2017"
  annotations_json: "data/coco/annotations/person_keypoints_val2017.json"

output:
  dir: "outputs"

benchmark:
  max_images: 100           # null for all images
  min_conf: 0.2             # Confidence threshold for visualization
  models:
    - mediapipe             # Currently implemented
    # - openpose            # Add when implemented
    # - mmpose              # Add when implemented
```

## Metrics Explained

**Per-Image Metrics** (`outputs/metrics/{model}.csv`):
- `detected_joints_count`: Joints with confidence ≥ threshold
- `mean_conf_all`: Average confidence across all 17 joints
- `mean_conf_detected`: Average confidence of detected joints only
- `valid_pose`: Boolean (≥8 joints detected)

**Leaderboard** (`outputs/leaderboard.csv`):
- `pose_rate`: Percentage of images with valid poses
- `mean_detected_joints`: Average detected joints per image
- `mean_conf`: Average confidence across all images

## Adding New Models

Implement the `PoseEstimator` interface in `src/pose_bench/models/`:

```python
from .base import PoseEstimator, PoseResult
import numpy as np

class MyModelEstimator(PoseEstimator):
    name = "mymodel"
    
    def __init__(self):
        # Initialize your model
        pass
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        # Run inference
        # Return COCO-17 keypoints (17, 2) and confidences (17,)
        return PoseResult(
            keypoints=np.array(...),  # (17, 2)
            conf=np.array(...),        # (17,)
            bbox=None,
            model_name=self.name,
        )
```

Register in `src/pose_bench/models/__init__.py`:

```python
from .mymodel import MyModelEstimator

MODEL_REGISTRY = {
    "mediapipe": MediaPipePoseEstimator,
    "mymodel": MyModelEstimator,  # Add here
}
```

## COCO-17 Keypoint Format

All models must output 17 keypoints in COCO order:

```
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
```

Skeleton edges are defined in `src/pose_bench/common/coco_schema.py`.

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Format code
black src/

# Type checking
mypy src/
```

## Performance Tips

- Start with `max_images: 100` for quick iteration
- Use `min_conf: 0.2` for visualization (lower shows more joints)
- MediaPipe runs ~20-30 images/sec on CPU, ~50+ on GPU (if available)
- For production benchmarks, use full dataset: `max_images: null`

## Limitations

- **Single-person focus**: Returns strongest detected person only
- **No ground truth comparison**: Metrics are detection-based, not accuracy-based
- **No GPU optimization**: MediaPipe uses CPU by default (GPU support varies)

## Future Enhancements

- [ ] Multi-person support with assignment to ground truth
- [ ] PCK/OKS metrics against COCO annotations
- [ ] GPU acceleration for models
- [ ] Video sequence support
- [ ] Real-time webcam demo mode

## License

MIT (or specify your license)

## Citation

If using COCO dataset:
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and others},
  booktitle={ECCV},
  year={2014}
}
```
