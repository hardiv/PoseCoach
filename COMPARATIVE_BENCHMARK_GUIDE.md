# Comparative Pose Estimation Benchmark - Usage Guide

Production-quality comparative benchmarking system for evaluating pose estimation models across multiple datasets (MPII baseline vs workout-specific images).

## 📋 Quick Start

### Full Comparison Pipeline (Recommended)
```bash
# Run everything: benchmark both datasets + analysis + visualizations
python run_full_comparison.py
```

This single command will:
1. Benchmark all models on MPII dataset
2. Benchmark all models on workout dataset
3. Generate comparison metrics and summary
4. Create 3 comparison visualizations
5. Display formatted results in terminal

**Time:** ~1-2 minutes (depending on dataset size)

---

## 🔧 Modular Usage

### 1. Run Comparative Benchmark

```bash
# Run both datasets
python run_comparative_benchmark.py --dataset both

# Run only MPII baseline
python run_comparative_benchmark.py --dataset mpii

# Run only workout images
python run_comparative_benchmark.py --dataset workout

# Use custom config
python run_comparative_benchmark.py --config my_config.yaml --dataset both
```

**Output:**
- `outputs/mpii/leaderboard.csv` - MPII results
- `outputs/workout/leaderboard.csv` - Workout results
- `outputs/leaderboard_mpii.csv` - Copied for comparison
- `outputs/leaderboard_workout.csv` - Copied for comparison

### 2. Generate Comparison Analysis

```bash
# Compare leaderboards and generate summary
python compare_datasets.py

# Custom paths
python compare_datasets.py --mpii outputs/leaderboard_mpii.csv \
                           --workout outputs/leaderboard_workout.csv \
                           --output outputs/comparison/my_summary.md
```

**Output:**
- `outputs/leaderboard_comparison.csv` - Full comparison metrics
- `outputs/comparison/comparison_summary.md` - Formatted markdown report

### 3. Generate Visualizations

```bash
# Create all comparison visualizations
python visualize_comparison.py

# Custom paths
python visualize_comparison.py --mpii-metrics outputs/mpii/metrics/mpii \
                               --workout-metrics outputs/workout/metrics/workout \
                               --output-dir outputs/figures
```

**Output:**
- `error_comparison_boxplot.png` - Side-by-side error distributions
- `error_delta_barchart.png` - Error increase per model (color-coded)
- `confidence_comparison_scatter.png` - Confidence drop visualization

### 4. Per-Joint Analysis (Bonus)

```bash
# Analyze which joints degrade most
python analyze_per_joint_comparison.py
```

**Output:**
- `outputs/comparison/per_joint_comparison.csv` - Per-joint error deltas
- `outputs/comparison/figures/per_joint_heatmap.png` - Heatmap visualization

---

## 📊 Understanding the Results

### Comparison Summary Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Δ Error (px)** | Pixel error increase on workout vs MPII | < 20px |
| **Δ Error (%)** | Percentage error increase | < 20% |
| **Conf Drop** | Confidence decrease | < 0.10 |
| **Conf Drop (%)** | Percentage confidence decrease | < 15% |

### Color Coding

**Error Delta Bar Chart:**
- 🟢 **Green** (< 20px increase): Good generalization
- 🟠 **Orange** (20-50px increase): Moderate degradation
- 🔴 **Red** (> 50px increase): Poor performance

**Confidence Scatter:**
- Below diagonal (y=x): Confidence dropped on workout images
- Near diagonal: Consistent confidence across datasets

---

## ⚙️ Configuration

### config_comparative.yaml

```yaml
datasets:
  mpii:
    name: "mpii"
    images_root: "data/mpii/images"
    annotations_json: "data/mpii/annotations/mpii_annotations.json"
    num_samples: 103  # -1 for all
    
  workout:
    name: "workout"
    images_root: "data/workout_images"
    annotations_json: "data/workout_images/annotations.json"
    num_samples: -1  # Use all available

output:
  dir: "outputs"
  comparison_dir: "outputs/comparison"

benchmark:
  min_conf: 0.2
  models:
    - mediapipe
    - yolov8-pose
    - movenet
```

### Adding a New Dataset

1. Create directory structure:
   ```
   data/my_dataset/
   ├── images/
   └── annotations.json
   ```

2. Add to `config_comparative.yaml`:
   ```yaml
   datasets:
     my_dataset:
       name: "my_dataset"
       images_root: "data/my_dataset/images"
       annotations_json: "data/my_dataset/annotations.json"
       num_samples: -1
   ```

3. Update scripts to include new dataset name

---

## 🔍 Error Handling

### Dataset Not Found
```
⚠ Dataset 'workout' not found at data/workout_images
  Skipping workout dataset.
```
**Solution:** Ensure dataset directory exists and contains images

### No Annotations
```
✗ Annotations file not found: data/workout_images/annotations.json
```
**Solution:** Create annotations or use COCO/MPII format annotations

### Empty Dataset
```
⚠ No images found in workout dataset at data/workout_images
  Skipping workout dataset.
```
**Solution:** Add images (jpg, png, etc.) to the directory

### Graceful Degradation
- If workout dataset missing → Runs only MPII
- If comparison fails → Benchmark results still saved
- If visualization fails → Analysis results still available

---

## 📂 Output Structure

```
outputs/
├── leaderboard_mpii.csv              # MPII results
├── leaderboard_workout.csv           # Workout results
├── leaderboard_comparison.csv        # Full comparison
├── comparison/
│   ├── comparison_summary.md         # Markdown report
│   ├── per_joint_comparison.csv      # Per-joint analysis
│   └── figures/
│       ├── error_comparison_boxplot.png
│       ├── error_delta_barchart.png
│       ├── confidence_comparison_scatter.png
│       └── per_joint_heatmap.png
├── mpii/
│   ├── leaderboard.csv
│   ├── metrics/
│   └── overlays/
└── workout/
    ├── leaderboard.csv
    ├── metrics/
    └── overlays/
```

---

## 💡 Tips for Supervisor Meeting

### Key Points to Highlight

1. **Modularity**: Each component runs independently
2. **Robustness**: Graceful error handling, works with missing datasets
3. **Production Quality**: Comprehensive documentation, help flags
4. **Visualization**: 4 types of plots for different insights
5. **Extensibility**: Easy to add new datasets/models

### Demo Flow

```bash
# 1. Show quick pipeline (if datasets ready)
python run_full_comparison.py

# 2. Or show modular approach
python run_comparative_benchmark.py --dataset both
python compare_datasets.py
python visualize_comparison.py

# 3. Show per-joint analysis (advanced)
python analyze_per_joint_comparison.py
```

### Talking Points

- **Side-by-side plots**: Show direct visual comparison with same scale
- **Color-coded delta chart**: Immediate understanding of model robustness
- **Confidence scatter**: Reveals which models maintain confidence
- **Markdown summary**: Automatic reporting with recommendations
- **Per-joint heatmap**: Identifies specific joint weaknesses

---

## 🚀 Next Steps

### For Production Use

1. **Collect Workout Images**: Add 30+ workout images to `data/workout_images/`
2. **Create Annotations**: Generate or manually create annotations
3. **Run Baseline**: `python run_full_comparison.py`
4. **Review Results**: Check `outputs/comparison/comparison_summary.md`
5. **Iterate**: Based on findings, select best model or collect more data

### For Development

1. **Add More Datasets**: Sports, outdoor, etc.
2. **Extend Metrics**: Add precision, recall, AUC
3. **Real Per-Joint Analysis**: Integrate actual keypoint predictions
4. **CI/CD Integration**: Automate benchmarking on new data

---

## ❓ Help & Troubleshooting

### Get Help
```bash
python run_comparative_benchmark.py --help
python compare_datasets.py --help
python visualize_comparison.py --help
python analyze_per_joint_comparison.py --help
```

### Common Issues

**Problem:** Models fail on workout images  
**Solution:** Check image format, size, ensure proper loading

**Problem:** Comparison shows huge increases  
**Solution:** Check if datasets are truly comparable, may need more workout data

**Problem:** Visualizations fail  
**Solution:** Ensure matplotlib, seaborn installed: `uv pip install matplotlib seaborn`

---

## 📝 Credits

Built for Fourth Year FYP, Imperial College London  
Date: February 2026  
System: Production-quality comparative benchmarking for supervisor meeting
