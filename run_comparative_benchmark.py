#!/usr/bin/env python3
"""
Run comparative pose estimation benchmark across multiple datasets.
Supports running MPII, workout, or both datasets independently.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dataclasses import dataclass

# Import existing benchmark functionality
from src.pose_bench.run_benchmark import run_benchmark
from src.pose_bench.config import Config, DatasetConfig, OutputConfig, BenchmarkConfig


@dataclass
class ComparativeConfig:
    """Configuration for comparative benchmarking."""
    datasets: Dict[str, Dict[str, Any]]
    output: Dict[str, str]
    benchmark: Dict[str, Any]


def load_comparative_config(config_path: str) -> ComparativeConfig:
    """Load comparative benchmark configuration from YAML."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return ComparativeConfig(**data)


def check_dataset_exists(dataset_config: Dict[str, Any], dataset_name: str) -> bool:
    """
    Check if a dataset exists and has images.
    
    Args:
        dataset_config: Dataset configuration dictionary
        dataset_name: Name of the dataset (for error messages)
        
    Returns:
        True if dataset is valid, False otherwise
    """
    images_root = Path(dataset_config['images_root'])
    
    if not images_root.exists():
        logging.warning(f"⚠ Dataset '{dataset_name}' not found at {images_root}")
        logging.warning(f"   Skipping {dataset_name} dataset.")
        return False
    
    # Check if directory has any images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(images_root.glob(f'**/*{ext}')))
    
    if not images:
        logging.warning(f"⚠ No images found in {dataset_name} dataset at {images_root}")
        logging.warning(f"   Skipping {dataset_name} dataset.")
        return False
    
    logging.info(f"✓ Found {len(images)} images in {dataset_name} dataset")
    return True


def run_dataset_benchmark(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
    base_output_dir: Path,
) -> bool:
    """
    Run benchmark on a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset-specific configuration
        benchmark_config: Benchmark configuration
        base_output_dir: Base output directory
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Running benchmark on {dataset_name.upper()} dataset")
    logging.info(f"{'='*80}")
    
    # Check if dataset exists
    if not check_dataset_exists(dataset_config, dataset_name):
        return False
    
    # Create Config object for this dataset
    dataset_cfg = DatasetConfig(
        name=dataset_config['name'],
        images_root=dataset_config['images_root'],
        annotations_json=dataset_config['annotations_json'],
    )
    
    output_cfg = OutputConfig(
        dir=str(base_output_dir / dataset_name)
    )
    
    # Determine max_images
    num_samples = dataset_config.get('num_samples', -1)
    max_images = None if num_samples == -1 else num_samples
    
    benchmark_cfg = BenchmarkConfig(
        max_images=max_images,
        min_conf=benchmark_config['min_conf'],
        models=benchmark_config['models'],
    )
    
    config = Config(
        dataset=dataset_cfg,
        output=output_cfg,
        benchmark=benchmark_cfg,
    )
    
    try:
        run_benchmark(config)
        
        # Copy leaderboard to comparison directory with dataset-specific name
        leaderboard_src = base_output_dir / dataset_name / "leaderboard.csv"
        leaderboard_dst = base_output_dir / f"leaderboard_{dataset_name}.csv"
        
        if leaderboard_src.exists():
            import shutil
            shutil.copy(leaderboard_src, leaderboard_dst)
            logging.info(f"✓ Saved {dataset_name} leaderboard to {leaderboard_dst}")
        
        return True
        
    except FileNotFoundError as e:
        logging.error(f"✗ Dataset error for {dataset_name}: {e}")
        logging.error(f"   Please ensure dataset is properly set up.")
        return False
    except Exception as e:
        logging.error(f"✗ Benchmark failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for comparative benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run comparative pose estimation benchmark across datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both datasets
  python run_comparative_benchmark.py --dataset both
  
  # Run only MPII
  python run_comparative_benchmark.py --dataset mpii
  
  # Run only workout images
  python run_comparative_benchmark.py --dataset workout
  
  # Use custom config
  python run_comparative_benchmark.py --config my_config.yaml --dataset both
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config_comparative.yaml",
        help="Path to comparative configuration YAML file (default: config_comparative.yaml)",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mpii", "workout", "both"],
        default="both",
        help="Which dataset(s) to run: mpii, workout, or both (default: both)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    
    # Load configuration
    try:
        config = load_comparative_config(args.config)
    except FileNotFoundError:
        logging.error(f"✗ Configuration file not found: {args.config}")
        logging.error(f"   Please create {args.config} or specify --config")
        sys.exit(1)
    except Exception as e:
        logging.error(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Determine which datasets to run
    datasets_to_run = []
    if args.dataset == "both":
        datasets_to_run = ["mpii", "workout"]
    else:
        datasets_to_run = [args.dataset]
    
    # Base output directory
    base_output_dir = Path(config.output['dir'])
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("="*80)
    logging.info("COMPARATIVE POSE ESTIMATION BENCHMARK")
    logging.info("="*80)
    logging.info(f"Datasets to run: {', '.join(datasets_to_run)}")
    logging.info(f"Models: {', '.join(config.benchmark['models'])}")
    logging.info(f"Output directory: {base_output_dir}")
    
    # Run benchmarks
    results = {}
    for dataset_name in datasets_to_run:
        if dataset_name not in config.datasets:
            logging.warning(f"⚠ Dataset '{dataset_name}' not found in config. Skipping.")
            continue
        
        dataset_config = config.datasets[dataset_name]
        success = run_dataset_benchmark(
            dataset_name,
            dataset_config,
            config.benchmark,
            base_output_dir,
        )
        results[dataset_name] = success
    
    # Summary
    logging.info(f"\n{'='*80}")
    logging.info("COMPARATIVE BENCHMARK SUMMARY")
    logging.info(f"{'='*80}")
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    if successful:
        logging.info(f"✓ Successfully completed: {', '.join(successful)}")
        logging.info(f"\nLeaderboard files:")
        for dataset_name in successful:
            leaderboard_path = base_output_dir / f"leaderboard_{dataset_name}.csv"
            if leaderboard_path.exists():
                logging.info(f"  - {leaderboard_path}")
    
    if failed:
        logging.warning(f"\n⚠ Failed or skipped: {', '.join(failed)}")
    
    if not successful:
        logging.error("\n✗ No datasets were successfully benchmarked.")
        sys.exit(1)
    
    logging.info(f"\n✓ Comparative benchmark complete!")
    logging.info(f"   Next steps:")
    logging.info(f"   - Run comparison analysis: python compare_datasets.py")
    logging.info(f"   - Generate visualizations: python visualize_comparison.py")
    logging.info(f"   - Or run full pipeline: python run_full_comparison.py")


if __name__ == "__main__":
    main()
