# PoseBench - Pose Estimation Benchmarking Suite

A focused benchmarking suite to answer: **"Which pose estimation model should we use for exercise form assessment, and when will it fail?"**

## 🎯 Experiment Goals

### Phase 1: Baseline Validation (COCO & MPII)
Establish baseline performance on standard datasets:
- **Purpose**: Sanity check - do models work? What's their general behavior?
- **Metrics**: Detection rates, confidence scores, inference speed, per-joint errors
- **Datasets**: COCO val2017 & MPII (150-300 images each)

### Phase 2: Exercise-Specific Testing (Real Gym Conditions)
Evaluate on real workout scenarios:
- **Purpose**: Answer "do these models work for MY problem?"
- **Analysis**: Visual inspection, consistency across viewpoints, robustness to lighting/occlusion
- **Dataset**: PA_WO (real gym exercises with varied capture conditions)
- **Key Questions**:
  - Which joints are unreliable? (avoid building form rules around them)
  - Which models are consistent? (low variance matters more than perfect accuracy)
  - What conditions cause failure? (viewpoint, occlusion, clothing, lighting)

## 🤖 Models

Three production-ready models:
- ✅ **MediaPipe BlazePose** - Fast, mobile-friendly, on-device inference
- ✅ **YOLOv8-Pose** - High accuracy, GPU-optimized
- ✅ **MoveNet** - Balanced speed/accuracy tradeoff

## 📁 Project Structure

```
PoseBench/
├── run_experiment.py              # 🚀 Main orchestrator - runs entire experiment
├── configs/                       # Phase-specific configurations
│   ├── coco_baseline.yaml         # Phase 1: COCO validation
│   ├── mpii_baseline.yaml         # Phase 1: MPII validation
│   └── gym_exercises.yaml         # Phase 2: Real gym testing
├── data/                          # Datasets
│   ├── coco/                      # COCO val2017
│   ├── mpii/                      # MPII dataset
│   └── pa_wo/                     # Gym exercise images
├── src/pose_bench/                # Modular benchmarking components
│   ├── inference.py               # 1️⃣ Run model predictions
│   ├── calculate_metrics.py       # 2️⃣ Compute aggregate metrics
│   ├── calculate_per_joint_errors.py  # 3️⃣ Per-joint error analysis
│   ├── generate_overlays.py       # 4️⃣ Create skeleton visualizations
│   ├── run_single_benchmark.py    # Orchestrate single model+dataset
│   ├── config.py                  # Configuration management
│   ├── common/                    # Shared utilities
│   │   ├── coco_schema.py         # COCO-17 keypoint schema
│   │   ├── io.py                  # Dataset loading
│   │   ├── draw_skeleton.py       # Skeleton visualization
│   │   └── metrics.py             # Metrics computation
│   ├── datasets/                  # Dataset adapters
│   │   └── mpii.py                # MPII adapter
│   └── models/                    # Pose estimators
│       ├── base.py                # Abstract interface
│       ├── mediapipe_blazepose.py
│       ├── yolov8_pose.py
│       └── movenet.py
├── scripts/
│   ├── download_coco.sh           # COCO dataset downloader
│   └── download_mpii.sh           # MPII dataset downloader
└── outputs/                       # Generated results
    ├── phase1_baseline/
    │   ├── coco/
    │   │   ├── predictions/       # Model predictions (JSON)
    │   │   ├── metrics/           # Aggregate & per-joint metrics (CSV)
    │   │   ├── overlays/          # Skeleton visualizations
    │   │   └── leaderboard.csv    # Model comparison
    │   └── mpii/
    └── phase2_gym/
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -e .

# Or using uv (faster)
uv pip install -e .
```

### 2. Download Datasets

```bash
# Phase 1: Baseline datasets
bash scripts/download_coco.sh    # COCO val2017 (~1.2GB)
bash scripts/download_mpii.sh    # MPII dataset

# Phase 2: Add gym exercise images to data/pa_wo/
```

### 3. Run Complete Experiment

```bash
# Run all phases end-to-end
python run_experiment.py
```

This will:
1. Run inference for all models on all datasets
2. Calculate comprehensive metrics
3. Generate per-joint error analysis (for datasets with ground truth)
4. Create skeleton overlay visualizations
5. Generate leaderboards for each dataset

### 4. Run Specific Phases

```bash
# Phase 1 only (baseline validation)
python run_experiment.py --configs configs/coco_baseline.yaml configs/mpii_baseline.yaml

# Phase 2 only (gym exercises)
python run_experiment.py --configs configs/gym_exercises.yaml

# Skip specific phases
python run_experiment.py --skip-phases coco
```

## 🔧 Modular Usage

Each component can be run independently:

### 1️⃣ Run Inference

```bash
python -m pose_bench.inference \
  --model mediapipe \
  --dataset-name coco \
  --images-root data/coco/val2017 \
  --annotations-json data/coco/annotations/person_keypoints_val2017.json \
  --output-dir outputs/test/predictions \
  --max-images 50
```

Output:
- `mediapipe_predictions.json` - Keypoints & confidences
- `mediapipe_inference_stats.json` - Timing statistics

### 2️⃣ Calculate Metrics

```bash
python -m pose_bench.calculate_metrics \
  --predictions outputs/test/predictions/mediapipe_predictions.json \
  --output-dir outputs/test/metrics \
  --min-conf 0.3
```

Output:
- `mediapipe_per_image_metrics.csv` - Per-image statistics
- `mediapipe_aggregate_metrics.json` - Overall performance

### 3️⃣ Calculate Per-Joint Errors

```bash
python -m pose_bench.calculate_per_joint_errors \
  --predictions outputs/test/predictions/mediapipe_predictions.json \
  --ground-truth outputs/test/predictions/ground_truth.json \
  --output-dir outputs/test/metrics \
  --min-conf 0.3
```

Output:
- `mediapipe_per_joint_errors.csv` - Joint-wise error statistics

### 4️⃣ Generate Overlays

```bash
python -m pose_bench.generate_overlays \
  --predictions outputs/test/predictions/mediapipe_predictions.json \
  --images-root data/coco/val2017 \
  --output-dir outputs/test/overlays/mediapipe \
  --min-conf 0.3
```

Output:
- Skeleton overlay images for each prediction

### 🎯 Run Single Model+Dataset Benchmark

```bash
python -m pose_bench.run_single_benchmark \
  --config configs/coco_baseline.yaml \
  --model mediapipe
```

Runs all steps (inference → metrics → errors → overlays) for one model.

## 📊 Output Structure

After running the experiment, outputs are organized by phase:

```
outputs/
├── phase1_baseline/
│   ├── coco/
│   │   ├── predictions/
│   │   │   ├── mediapipe_predictions.json
│   │   │   ├── yolov8-pose_predictions.json
│   │   │   ├── movenet_predictions.json
│   │   │   └── ground_truth.json
│   │   ├── metrics/
│   │   │   ├── mediapipe_per_image_metrics.csv
│   │   │   ├── mediapipe_aggregate_metrics.json
│   │   │   ├── mediapipe_per_joint_errors.csv
│   │   │   └── ... (same for other models)
│   │   ├── overlays/
│   │   │   ├── mediapipe/coco/
│   │   │   ├── yolov8-pose/coco/
│   │   │   └── movenet/coco/
│   │   └── leaderboard.csv               # 📈 Model comparison
│   └── mpii/
│       └── ... (same structure)
└── phase2_gym/
    └── ... (same structure)
```

## 📈 Key Metrics

### Aggregate Metrics
- **Pose Detection Rate**: % images with valid pose (≥8 joints detected)
- **Mean Detected Joints**: Average joints detected per image
- **Mean Confidence**: Average confidence across all predictions
- **Inference Time**: Mean & std deviation (ms)

### Per-Joint Metrics (when ground truth available)
- **Detection Rate**: % visible joints detected
- **Mean Pixel Error**: Average distance from ground truth
- **Median Pixel Error**: Robust error measure
- **Error Std Deviation**: Consistency measure

## 🔍 Analysis Guidelines

### Phase 1: Baseline Validation
Look for:
- Overall detection rates (should be >80% on standard datasets)
- Inference speed (MediaPipe ~20-50ms, YOLOv8 ~30-100ms, MoveNet ~40-80ms)
- Per-joint reliability (which joints have high error or low detection?)

### Phase 2: Gym Exercise Testing
Focus on:
- **Visual Inspection**: Do overlays look correct across different viewpoints?
- **Consistency**: Does same model produce similar results across angles?
- **Failure Modes**: Which conditions cause breakdown?
  - Occlusion (baggy clothing, equipment)
  - Lighting (dim gym lighting)
  - Viewpoint (front vs side vs 45°)
  - Body types

## 🎨 Configuration

Edit config files in `configs/` to customize:

```yaml
dataset:
  name: "coco"                              # Dataset identifier
  images_root: "data/coco/val2017"          # Path to images
  annotations_json: "data/coco/..."         # Annotations (optional)

output:
  dir: "outputs/phase1_baseline/coco"       # Output directory

benchmark:
  max_images: 150                           # Limit for quick tests (null = all)
  min_conf: 0.3                             # Confidence threshold
  models:                                   # Models to evaluate
    - mediapipe
    - yolov8-pose
    - movenet
```

## 🛠️ Development

### Adding a New Model

1. Create `src/pose_bench/models/your_model.py`:

```python
from .base import PoseEstimator, PoseResult
import numpy as np

class YourModelEstimator(PoseEstimator):
    name = "your_model"
    
    def __init__(self):
        # Initialize model
        pass
    
    def predict(self, bgr_image: np.ndarray) -> PoseResult:
        # Run inference, return COCO-17 keypoints
        return PoseResult(
            keypoints=np.array(...),  # (17, 2)
            conf=np.array(...),        # (17,)
            bbox=None,
            model_name=self.name,
        )
```

2. Register in `src/pose_bench/models/__init__.py`:

```python
MODEL_REGISTRY = {
    "mediapipe": MediaPipePoseEstimator,
    "yolov8-pose": YOLOv8PoseEstimator,
    "movenet": MoveNetEstimator,
    "your_model": YourModelEstimator,  # Add here
}
```

3. Add to config file `models` list

### Adding a New Dataset

1. Create adapter in `src/pose_bench/datasets/`
2. Implement loading and annotation parsing
3. Map keypoints to COCO-17 schema
4. Create config file in `configs/`

## 📦 Dependencies

Core:
- `opencv-python` - Image I/O and visualization
- `numpy` - Array operations
- `pandas` - Metrics and CSV handling
- `tqdm` - Progress bars
- `pyyaml` - Configuration

Models:
- `mediapipe` - BlazePose
- `ultralytics` - YOLOv8
- `tensorflow` / `tensorflow-hub` - MoveNet

## 🔗 COCO-17 Keypoint Format

All models output 17 keypoints in COCO order:

```
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
```

## 📝 License

MIT

## 📚 Citation

If using COCO dataset:
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and others},
  booktitle={ECCV},
  year={2014}
}
```

If using MPII dataset:
```bibtex
@inproceedings{andriluka20142d,
  title={2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  author={Andriluka, Mykhaylo and Pishchulin, Leonid and Gehler, Peter and Schiele, Bernt},
  booktitle={CVPR},
  year={2014}
}
```
