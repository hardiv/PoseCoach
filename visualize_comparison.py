#!/usr/bin/env python3
"""
Generate comparative visualizations for pose estimation benchmark results.
Creates side-by-side plots, delta charts, and confidence comparisons.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')


def load_per_image_metrics(metrics_dir: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load per-image metrics from all model CSV files.
    
    Args:
        metrics_dir: Directory containing model CSV files
        dataset_name: Name of the dataset
        
    Returns:
        Combined DataFrame with all per-image results
    """
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
    
    csv_files = list(metrics_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {metrics_dir}")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset_name
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_side_by_side_error_boxplots(
    df_mpii: pd.DataFrame,
    df_workout: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (14, 6),
) -> None:
    """
    Create side-by-side box plots comparing error distributions.
    
    Args:
        df_mpii: MPII per-image metrics
        df_workout: Workout per-image metrics
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    # Check if error column exists
    if "mean_pixel_error" not in df_mpii.columns or "mean_pixel_error" not in df_workout.columns:
        print("⚠ Pixel error data not found in one or both datasets. Skipping box plot.")
        return
    
    # Filter out NaN errors
    df_mpii_clean = df_mpii.dropna(subset=["mean_pixel_error"])
    df_workout_clean = df_workout.dropna(subset=["mean_pixel_error"])
    
    # Get model names (sorted for consistency)
    models = sorted(df_mpii_clean["model_name"].unique())
    
    # Prepare data for both datasets
    mpii_data = [df_mpii_clean[df_mpii_clean["model_name"] == model]["mean_pixel_error"].values
                 for model in models]
    workout_data = [df_workout_clean[df_workout_clean["model_name"] == model]["mean_pixel_error"].values
                    for model in models]
    
    # Determine common y-axis scale
    all_errors = np.concatenate([np.concatenate(mpii_data), np.concatenate(workout_data)])
    y_min, y_max = all_errors.min() * 0.9, all_errors.max() * 1.1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # MPII box plot (left)
    bp1 = ax1.boxplot(
        mpii_data,
        tick_labels=models,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
        widths=0.6,
    )
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lavender']
    for patch, color in zip(bp1['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Per-Image Pixel Error (px)', fontsize=11, fontweight='bold')
    ax1.set_title('MPII Dataset (Baseline)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.set_ylim(y_min, y_max)
    
    # Workout box plot (right)
    bp2 = ax2.boxplot(
        workout_data,
        tick_labels=models,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
        widths=0.6,
    )
    
    # Color boxes (same colors as MPII)
    for patch, color in zip(bp2['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_title('Workout Dataset', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add overall title
    fig.suptitle('Error Distribution Comparison: MPII vs Workout', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=6, label='Mean'),
        Line2D([0], [0], color='orange', linewidth=2, label='Median'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.98), framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved side-by-side error comparison to {output_path}")


def plot_error_delta_barchart(
    comparison_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Create bar chart showing error increase (workout - MPII).
    
    Args:
        comparison_df: Comparison DataFrame with error_delta_px column
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if "error_delta_px" not in comparison_df.columns:
        print("⚠ error_delta_px not found in comparison data. Skipping delta chart.")
        return
    
    # Sort by error delta
    df_sorted = comparison_df.sort_values("error_delta_px")
    
    # Assign colors based on thresholds
    colors = []
    for delta in df_sorted["error_delta_px"]:
        if delta < 20:
            colors.append('green')
        elif delta < 50:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    x_pos = np.arange(len(df_sorted))
    bars = ax.bar(
        x_pos,
        df_sorted["error_delta_px"],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5,
    )
    
    # Add value labels on bars
    for i, (bar, delta) in enumerate(zip(bars, df_sorted["error_delta_px"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2 if height > 0 else height - 2,
            f'{delta:.1f}px',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=10,
            fontweight='bold',
        )
    
    # Labels and title
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Increase (px)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Pose Error Increase: Workout vs MPII Baseline',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted["model_name"])
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Good (< 20px)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='Moderate (20-50px)'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Poor (> 50px)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error delta bar chart to {output_path}")


def plot_confidence_comparison_scatter(
    comparison_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (8, 8),
) -> None:
    """
    Create scatter plot comparing confidence between datasets.
    
    Args:
        comparison_df: Comparison DataFrame with confidence columns
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if "mean_conf_mpii" not in comparison_df.columns or "mean_conf_workout" not in comparison_df.columns:
        print("⚠ Confidence data not found in comparison. Skipping scatter plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot diagonal line (y=x, no change)
    min_conf = min(comparison_df["mean_conf_mpii"].min(), comparison_df["mean_conf_workout"].min())
    max_conf = max(comparison_df["mean_conf_mpii"].max(), comparison_df["mean_conf_workout"].max())
    margin = (max_conf - min_conf) * 0.1
    
    ax.plot([min_conf - margin, max_conf + margin], 
            [min_conf - margin, max_conf + margin],
            'k--', alpha=0.5, linewidth=2, label='No change (y=x)')
    
    # Color code by confidence drop
    colors = []
    for _, row in comparison_df.iterrows():
        conf_drop = row['mean_conf_mpii'] - row['mean_conf_workout']
        if conf_drop < 0.05:
            colors.append('green')
        elif conf_drop < 0.15:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Scatter plot
    scatter = ax.scatter(
        comparison_df["mean_conf_mpii"],
        comparison_df["mean_conf_workout"],
        c=colors,
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=2,
    )
    
    # Add model name labels
    for _, row in comparison_df.iterrows():
        ax.annotate(
            row["model_name"],
            (row["mean_conf_mpii"], row["mean_conf_workout"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
        )
    
    # Labels and title
    ax.set_xlabel('MPII Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Workout Confidence', fontsize=12, fontweight='bold')
    ax.set_title(
        'Confidence Comparison: MPII vs Workout\n(Points below line = confidence dropped)',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(min_conf - margin, max_conf + margin)
    ax.set_ylim(min_conf - margin, max_conf + margin)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        ax.lines[0],  # Diagonal line
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Small drop (< 0.05)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='Moderate drop (0.05-0.15)'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Large drop (> 0.15)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confidence comparison scatter to {output_path}")


def main():
    """Main entry point for comparative visualization."""
    parser = argparse.ArgumentParser(
        description="Generate comparative visualizations for pose estimation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations with default paths
  python visualize_comparison.py
  
  # Specify custom paths
  python visualize_comparison.py --mpii-metrics outputs/mpii/metrics/mpii \\
                                 --workout-metrics outputs/workout/metrics/workout
  
  # Custom output directory
  python visualize_comparison.py --output-dir outputs/comparison/figures
        """
    )
    
    parser.add_argument(
        "--mpii-metrics",
        type=str,
        default="outputs/mpii/metrics/mpii",
        help="Path to MPII per-image metrics directory",
    )
    
    parser.add_argument(
        "--workout-metrics",
        type=str,
        default="outputs/workout/metrics/workout",
        help="Path to workout per-image metrics directory",
    )
    
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="outputs/leaderboard_comparison.csv",
        help="Path to comparison CSV (from compare_datasets.py)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison/figures",
        help="Directory to save visualizations",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARATIVE VISUALIZATION GENERATION")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load per-image metrics
    print("\n📂 Loading per-image metrics...")
    try:
        df_mpii = load_per_image_metrics(Path(args.mpii_metrics), "mpii")
        print(f"   ✓ Loaded {len(df_mpii)} MPII records from {df_mpii['model_name'].nunique()} models")
    except FileNotFoundError as e:
        print(f"✗ Error loading MPII metrics: {e}")
        print(f"  Run comparative benchmark first: python run_comparative_benchmark.py")
        sys.exit(1)
    
    try:
        df_workout = load_per_image_metrics(Path(args.workout_metrics), "workout")
        print(f"   ✓ Loaded {len(df_workout)} workout records from {df_workout['model_name'].nunique()} models")
    except FileNotFoundError as e:
        print(f"✗ Error loading workout metrics: {e}")
        print(f"  Run comparative benchmark first: python run_comparative_benchmark.py")
        sys.exit(1)
    
    # Load comparison CSV
    print(f"\n📂 Loading comparison data...")
    try:
        comparison_df = pd.read_csv(args.comparison_csv)
        print(f"   ✓ Loaded comparison for {len(comparison_df)} models")
    except FileNotFoundError as e:
        print(f"✗ Error loading comparison CSV: {e}")
        print(f"  Run comparison analysis first: python compare_datasets.py")
        sys.exit(1)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    # 1. Side-by-side box plots
    print("\n  1/3: Creating side-by-side error box plots...")
    plot_side_by_side_error_boxplots(
        df_mpii,
        df_workout,
        output_dir / "error_comparison_boxplot.png",
    )
    
    # 2. Delta bar chart
    print("  2/3: Creating error delta bar chart...")
    plot_error_delta_barchart(
        comparison_df,
        output_dir / "error_delta_barchart.png",
    )
    
    # 3. Confidence scatter
    print("  3/3: Creating confidence comparison scatter...")
    plot_confidence_comparison_scatter(
        comparison_df,
        output_dir / "confidence_comparison_scatter.png",
    )
    
    print(f"\n{'='*80}")
    print("✓ All visualizations complete!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - {output_dir / 'error_comparison_boxplot.png'}")
    print(f"  - {output_dir / 'error_delta_barchart.png'}")
    print(f"  - {output_dir / 'confidence_comparison_scatter.png'}")
    print(f"\nView figures: open {output_dir}")


if __name__ == "__main__":
    main()
