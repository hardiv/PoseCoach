#!/usr/bin/env python3
"""
PoseBench - Pose Estimation Benchmarking Suite

Usage:
    python run_benchmark.py --config configs/coco_baseline.yaml
    python run_benchmark.py --config configs/mpii_baseline.yaml
    python run_benchmark.py --config configs/gym_exercises.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_bench.run_benchmark import main

if __name__ == "__main__":
    main()
