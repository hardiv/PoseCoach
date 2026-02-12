#!/usr/bin/env python3
"""
Compare pose estimation benchmark results across datasets.
Generates comparison metrics and markdown summary.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime


def load_leaderboard(filepath: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load leaderboard CSV and add dataset column.
    
    Args:
        filepath: Path to leaderboard CSV
        dataset_name: Name of the dataset
        
    Returns:
        DataFrame with dataset column added
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Leaderboard not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['dataset'] = dataset_name
    return df


def compute_comparison_metrics(
    df_mpii: pd.DataFrame,
    df_workout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute comparison metrics between two datasets.
    
    Args:
        df_mpii: MPII leaderboard DataFrame
        df_workout: Workout leaderboard DataFrame
        
    Returns:
        DataFrame with comparison metrics
    """
    # Merge on model_name
    comparison = pd.merge(
        df_mpii,
        df_workout,
        on='model_name',
        suffixes=('_mpii', '_workout')
    )
    
    # Compute deltas and percentages
    comparison['error_delta_px'] = (
        comparison['mean_pixel_error_workout'] - comparison['mean_pixel_error_mpii']
    )
    
    comparison['error_pct_increase'] = (
        (comparison['mean_pixel_error_workout'] - comparison['mean_pixel_error_mpii']) /
        comparison['mean_pixel_error_mpii'] * 100
    )
    
    comparison['conf_drop'] = (
        comparison['mean_conf_mpii'] - comparison['mean_conf_workout']
    )
    
    comparison['conf_drop_pct'] = (
        comparison['conf_drop'] / comparison['mean_conf_mpii'] * 100
    )
    
    # Check inference time consistency
    if 'mean_inference_ms_mpii' in comparison.columns and 'mean_inference_ms_workout' in comparison.columns:
        comparison['inference_time_delta_ms'] = (
            comparison['mean_inference_ms_workout'] - comparison['mean_inference_ms_mpii']
        )
        comparison['inference_time_delta_pct'] = (
            comparison['inference_time_delta_ms'] / comparison['mean_inference_ms_mpii'] * 100
        )
    
    return comparison


def generate_markdown_summary(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate markdown summary of comparison.
    
    Args:
        comparison: Comparison DataFrame
        output_path: Path to save markdown file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# Pose Estimation: MPII vs Workout Dataset Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Full comparison table
        f.write("## 📊 Full Comparison Table\n\n")
        f.write("| Model | MPII Error (px) | Workout Error (px) | Δ Error (px) | Δ Error (%) | MPII Conf | Workout Conf | Conf Drop | Conf Drop (%) |\n")
        f.write("|-------|-----------------|--------------------|--------------|--------------|-----------|--------------|-----------|--------------|\n")
        
        for _, row in comparison.iterrows():
            f.write(
                f"| {row['model_name']} | "
                f"{row['mean_pixel_error_mpii']:.2f} | "
                f"{row['mean_pixel_error_workout']:.2f} | "
                f"{row['error_delta_px']:.2f} | "
                f"{row['error_pct_increase']:.1f}% | "
                f"{row['mean_conf_mpii']:.3f} | "
                f"{row['mean_conf_workout']:.3f} | "
                f"{row['conf_drop']:.3f} | "
                f"{row['conf_drop_pct']:.1f}% |\n"
            )
        
        f.write("\n")
        
        # Performance timing comparison
        if 'inference_time_delta_ms' in comparison.columns:
            f.write("## ⏱️ Inference Time Consistency\n\n")
            f.write("| Model | MPII (ms) | Workout (ms) | Δ Time (ms) | Δ Time (%) |\n")
            f.write("|-------|-----------|--------------|-------------|------------|\n")
            
            for _, row in comparison.iterrows():
                f.write(
                    f"| {row['model_name']} | "
                    f"{row['mean_inference_ms_mpii']:.2f} | "
                    f"{row['mean_inference_ms_workout']:.2f} | "
                    f"{row['inference_time_delta_ms']:.2f} | "
                    f"{row['inference_time_delta_pct']:.1f}% |\n"
                )
            
            f.write("\n")
        
        # Key findings
        f.write("## 🔍 Key Findings\n\n")
        
        # Best and worst performers
        best_model = comparison.loc[comparison['error_delta_px'].idxmin()]
        worst_model = comparison.loc[comparison['error_delta_px'].idxmax()]
        
        f.write(f"### Model Robustness (Smallest Error Increase)\n")
        f.write(f"**🏆 Most Robust:** {best_model['model_name']}\n")
        f.write(f"- Error increase: {best_model['error_delta_px']:.2f}px ({best_model['error_pct_increase']:.1f}%)\n")
        f.write(f"- Confidence drop: {best_model['conf_drop']:.3f} ({best_model['conf_drop_pct']:.1f}%)\n\n")
        
        f.write(f"**⚠️ Most Degraded:** {worst_model['model_name']}\n")
        f.write(f"- Error increase: {worst_model['error_delta_px']:.2f}px ({worst_model['error_pct_increase']:.1f}%)\n")
        f.write(f"- Confidence drop: {worst_model['conf_drop']:.3f} ({worst_model['conf_drop_pct']:.1f}%)\n\n")
        
        # Confidence analysis
        most_conf_drop = comparison.loc[comparison['conf_drop'].idxmax()]
        least_conf_drop = comparison.loc[comparison['conf_drop'].idxmin()]
        
        f.write(f"### Confidence Analysis\n")
        f.write(f"**Largest confidence drop:** {most_conf_drop['model_name']} "
                f"(-{most_conf_drop['conf_drop']:.3f}, -{most_conf_drop['conf_drop_pct']:.1f}%)\n")
        f.write(f"**Smallest confidence drop:** {least_conf_drop['model_name']} "
                f"(-{least_conf_drop['conf_drop']:.3f}, -{least_conf_drop['conf_drop_pct']:.1f}%)\n\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        
        # Categorize models
        low_degradation = comparison[comparison['error_pct_increase'] < 20]
        medium_degradation = comparison[
            (comparison['error_pct_increase'] >= 20) &
            (comparison['error_pct_increase'] < 50)
        ]
        high_degradation = comparison[comparison['error_pct_increase'] >= 50]
        
        if not low_degradation.empty:
            f.write(f"### ✅ Recommended for Workout Images (< 20% error increase)\n")
            for _, row in low_degradation.iterrows():
                f.write(f"- **{row['model_name']}**: "
                       f"{row['error_pct_increase']:.1f}% increase, "
                       f"maintains good accuracy\n")
            f.write("\n")
        
        if not medium_degradation.empty:
            f.write(f"### ⚠️ Moderate Performance (20-50% error increase)\n")
            for _, row in medium_degradation.iterrows():
                f.write(f"- **{row['model_name']}**: "
                       f"{row['error_pct_increase']:.1f}% increase, "
                       f"may need tuning\n")
            f.write("\n")
        
        if not high_degradation.empty:
            f.write(f"### ❌ Poor Performance (> 50% error increase)\n")
            for _, row in high_degradation.iterrows():
                f.write(f"- **{row['model_name']}**: "
                       f"{row['error_pct_increase']:.1f}% increase, "
                       f"not suitable for workout images\n")
            f.write("\n")
        
        # Overall summary
        f.write("## 📝 Summary\n\n")
        avg_error_increase = comparison['error_pct_increase'].mean()
        avg_conf_drop = comparison['conf_drop_pct'].mean()
        
        f.write(f"- **Average error increase:** {avg_error_increase:.1f}%\n")
        f.write(f"- **Average confidence drop:** {avg_conf_drop:.1f}%\n")
        f.write(f"- **Models tested:** {len(comparison)}\n")
        
        if avg_error_increase < 30:
            f.write(f"\n✅ **Overall:** Models show good generalization to workout images.\n")
        elif avg_error_increase < 60:
            f.write(f"\n⚠️ **Overall:** Models show moderate degradation on workout images. Consider domain-specific training.\n")
        else:
            f.write(f"\n❌ **Overall:** Significant degradation on workout images. Domain adaptation strongly recommended.\n")
    
    print(f"✓ Saved comparison summary to {output_path}")


def main():
    """Main entry point for dataset comparison."""
    parser = argparse.ArgumentParser(
        description="Compare pose estimation benchmark results across datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with default paths
  python compare_datasets.py
  
  # Specify custom paths
  python compare_datasets.py --mpii results/leaderboard_mpii.csv --workout results/leaderboard_workout.csv
  
  # Custom output location
  python compare_datasets.py --output results/my_comparison.md
        """
    )
    
    parser.add_argument(
        "--mpii",
        type=str,
        default="outputs/leaderboard_mpii.csv",
        help="Path to MPII leaderboard CSV (default: outputs/leaderboard_mpii.csv)",
    )
    
    parser.add_argument(
        "--workout",
        type=str,
        default="outputs/leaderboard_workout.csv",
        help="Path to workout leaderboard CSV (default: outputs/leaderboard_workout.csv)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/comparison/comparison_summary.md",
        help="Path to save comparison summary (default: outputs/comparison/comparison_summary.md)",
    )
    
    parser.add_argument(
        "--csv-output",
        type=str,
        default="outputs/leaderboard_comparison.csv",
        help="Path to save comparison CSV (default: outputs/leaderboard_comparison.csv)",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATASET COMPARISON ANALYSIS")
    print("="*80)
    
    # Load leaderboards
    try:
        print(f"\n📂 Loading MPII leaderboard from {args.mpii}...")
        df_mpii = load_leaderboard(Path(args.mpii), "mpii")
        print(f"   ✓ Loaded {len(df_mpii)} models")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"  Run comparative benchmark first: python run_comparative_benchmark.py")
        sys.exit(1)
    
    try:
        print(f"\n📂 Loading workout leaderboard from {args.workout}...")
        df_workout = load_leaderboard(Path(args.workout), "workout")
        print(f"   ✓ Loaded {len(df_workout)} models")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"  Run comparative benchmark first: python run_comparative_benchmark.py")
        sys.exit(1)
    
    # Compute comparison metrics
    print(f"\n🔬 Computing comparison metrics...")
    comparison = compute_comparison_metrics(df_mpii, df_workout)
    print(f"   ✓ Analyzed {len(comparison)} models")
    
    # Save comparison CSV
    csv_output_path = Path(args.csv_output)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_output_path, index=False)
    print(f"   ✓ Saved comparison data to {csv_output_path}")
    
    # Generate markdown summary
    print(f"\n📝 Generating comparison summary...")
    generate_markdown_summary(comparison, Path(args.output))
    
    # Print quick summary to console
    print(f"\n{'='*80}")
    print("QUICK SUMMARY")
    print(f"{'='*80}")
    
    for _, row in comparison.iterrows():
        symbol = "✅" if row['error_pct_increase'] < 20 else "⚠️" if row['error_pct_increase'] < 50 else "❌"
        print(f"{symbol} {row['model_name']:15s} | "
              f"Error: +{row['error_delta_px']:6.2f}px (+{row['error_pct_increase']:5.1f}%) | "
              f"Conf drop: -{row['conf_drop']:.3f}")
    
    print(f"\n✓ Comparison complete!")
    print(f"  View full report: {args.output}")


if __name__ == "__main__":
    main()
