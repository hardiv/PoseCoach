#!/usr/bin/env python3
"""
Run complete pose estimation experiment end-to-end.

This orchestrator runs:
- Phase 1: Baseline validation on COCO and MPII datasets
- Phase 2: Exercise-specific testing on gym dataset

For each phase and dataset:
1. Run inference for all models
2. Calculate metrics
3. Generate per-joint error analysis
4. Create skeleton overlay visualizations  
5. Generate leaderboard
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_bench.config import Config, DatasetConfig, OutputConfig, BenchmarkConfig
from pose_bench.run_single_benchmark import run_single_benchmark
from pose_bench.calculate_metrics import create_leaderboard


# Define experiment phases directly in Python
EXPERIMENT_PHASES = [
    # Phase 1: COCO Baseline
    Config(
        dataset=DatasetConfig(
            name="coco",
            images_root="data/coco/val2017",
            annotations_json="data/coco/annotations/person_keypoints_val2017.json",
        ),
        output=OutputConfig(dir="outputs/phase1_baseline/coco"),
        benchmark=BenchmarkConfig(
            max_images=150,  # Subset for baseline
            min_conf=0.3,
            models=["mediapipe", "yolov8-pose", "movenet"],
        ),
    ),
    # Phase 1: MPII Baseline
    Config(
        dataset=DatasetConfig(
            name="mpii",
            images_root="data/mpii/images",
            annotations_json="data/mpii/annotations/mpii_annotations.json",
        ),
        output=OutputConfig(dir="outputs/phase1_baseline/mpii"),
        benchmark=BenchmarkConfig(
            max_images=150,  # Subset for baseline
            min_conf=0.3,
            models=["mediapipe", "yolov8-pose", "movenet"],
        ),
    ),
    # Phase 2: Gym Exercises
    Config(
        dataset=DatasetConfig(
            name="gym_exercises",
            images_root="data/pa_wo",
            annotations_json=None,  # No annotations for gym data
        ),
        output=OutputConfig(dir="outputs/phase2_gym"),
        benchmark=BenchmarkConfig(
            max_images=None,  # All images
            min_conf=0.2,  # Lower threshold for challenging conditions
            models=["mediapipe", "yolov8-pose", "movenet"],
        ),
    ),
]


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    banner = f"\n{char * width}\n{text.center(width)}\n{char * width}\n"
    print(banner)
    logging.info(banner)


def run_experiment(
    phases: List[int] = None,
    skip_phases: List[str] = None,
) -> None:
    """
    Run complete experiment across all phases.
    
    Args:
        phases: List of phase indices to run (1-3), None = all
        skip_phases: List of phase names to skip (e.g., ["coco", "mpii"])
    """
    skip_phases = skip_phases or []
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("outputs") / f"experiment_{timestamp}.log"
    logger = setup_logging(log_file)
    
    print_banner("POSE ESTIMATION EXPERIMENT", "=")
    logger.info(f"Experiment started at {datetime.now()}")
    logger.info(f"Total phases: {len(EXPERIMENT_PHASES)}")
    logger.info(f"Log file: {log_file}")
    
    overall_results = []
    
    # Filter phases if specified
    configs_to_run = EXPERIMENT_PHASES
    if phases:
        configs_to_run = [EXPERIMENT_PHASES[i-1] for i in phases if 1 <= i <= len(EXPERIMENT_PHASES)]
        logger.info(f"Running phases: {phases}")
    
    # Process each phase
    for config_idx, config in enumerate(configs_to_run, 1):
        phase_name = config.dataset.name
        
        # Check if phase should be skipped
        if phase_name in skip_phases:
            logger.info(f"Skipping phase: {phase_name}")
            continue
        
        print_banner(f"PHASE {config_idx}/{len(configs)}: {phase_name.upper()}", "=")
        logger.info(f"Config: {config_path}")
        logger.info(f"Dataset: {config.dataset.name}")
        logger.info(f"Models: {', '.join(config.benchmark.models)}")
        logger.info(f"Max images: {config.benchmark.max_images or 'all'}")
        
        phase_results = {
            "phase": phase_name,_to_run)}: {phase_name.upper()}", "=")
        logger.info(f"Images: {config.dataset.images_root
        }
        
        # Run benchmark for each model
        for model_idx, model_name in enumerate(config.benchmark.models, 1):
            logger.info(f"\n{'─' * 80}")
            logger.info(f"Model {model_idx}/{len(config.benchmark.models)}: {model_name}")
            logger.info(f"{'─' * 80}")
            
            try:
                success = run_single_benchmark(model_name, config)
                phase_results["models"][model_name] = "success" if success else "failed"
                
                if not success:
                    logger.error(f"Benchmark failed for {model_name}")
                    
            except Exception as e:
                logger.error(f"Error running benchmark for {model_name}: {e}")
                phase_results["models"][model_name] = f"error: {e}"
                continue
        
        # Generate leaderboard for this phase
        try:
            logger.info(f"\nGenerating leaderboard for {phase_name}...")
            metrics_dir = config.get_output_dir() / "metrics"
            output_dir = config.get_output_dir()
            create_leaderboard(metrics_dir, output_dir)
            logger.info(f"✓ Leaderboard created\n")
        except Exception as e:
            logger.error(f"Failed to create leaderboard: {e}")
        
        overall_results.append(phase_results)
        
        print_banner(f"PHASE {config_idx} COMPLETE: {phase_name.upper()}", "=")
    
    # Print experiment summary
    print_banner("EXPERIMENT COMPLETE", "=")
    logger.info("\nSummary:")
    
    for phase_result in overall_results:
        phase_name = phase_result["phase"]
        logger.info(f"\n  {phase_name}:")
        for model_name, status in phase_result["models"].items():
            status_symbol = "✓" if status == "success" else "✗"
            logger.info(f"    {status_symbol} {model_name}: {status}")
    
    logger.info(f"\nExperiment log saved to: {log_file}")
    logger.info(f"Results saved to: outputs/")
    
    print_banner("ALL DONE!", "=")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete pose estimation experiment end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment (all phases)
  python run_experiment.py3 phases)
  python3 run_experiment.py
  
  # Run specific phases (1=COCO, 2=MPII, 3=Gym)
  python3 run_experiment.py --phases 1 2
  
  # Skip specific phases by name
  python3 run_experiment.py --skip-phases coco
  
  # Run Phase 1 only (baseline validation)
  python3 run_experiment.py --phases 1 2
  
  # Run Phase 2 only (gym exercises)
  python3 run_experiment.py --phases 3
        """,
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=None,
        help="Phase numbers to run: 1=COCO, 2=MPII, 3=Gym (default: all)",
    )
    parser.add_argument(
        "--skip-phases",
        type=str,
        nargs="+",
        default=[],
        help="Phase names to skip (e.g., coco mpii gym_exercises)",
    )
    
    args = parser.parse_args()
    
    try:
        run_experiment(
            phases=args.phaseip_phases,
        )
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        logging.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
