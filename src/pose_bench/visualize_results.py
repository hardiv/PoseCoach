"""Visualization script for pose estimation benchmark results."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def load_per_image_metrics(metrics_dir: Path) -> pd.DataFrame:
    """
    Load per-image metrics from all model CSV files.
    
    Args:
        metrics_dir: Directory containing model CSV files
        
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
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def plot_error_distribution_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    dataset_name: str = "MPII",
    figsize: tuple = (10, 6),
) -> None:
    """
    Create a box plot showing error distribution across models.
    
    Args:
        df: DataFrame with per-image metrics including 'mean_pixel_error' and 'model_name'
        output_path: Path to save the figure
        dataset_name: Name of the dataset for title
        figsize: Figure size (width, height)
    """
    # Check if error column exists
    if "mean_pixel_error" not in df.columns:
        raise ValueError(
            "Column 'mean_pixel_error' not found. "
            "This visualization requires ground truth data (e.g., MPII dataset)."
        )
    
    # Filter out NaN errors
    df_clean = df.dropna(subset=["mean_pixel_error"])
    
    if df_clean.empty:
        raise ValueError("No valid error data found after removing NaN values.")
    
    # Get unique model names and sort them
    model_names = sorted(df_clean["model_name"].unique())
    
    # Prepare data for box plot
    error_data = [
        df_clean[df_clean["model_name"] == model]["mean_pixel_error"].values
        for model in model_names
    ]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    bp = ax.boxplot(
        error_data,
        tick_labels=model_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8),
        widths=0.6,
    )
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lavender']
    for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean and std annotations
    for i, (model, data) in enumerate(zip(model_names, error_data), 1):
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Position text above the box plot
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(
            i,
            y_pos,
            f'μ={mean_val:.1f}px\nσ={std_val:.1f}px',
            ha='center',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
    
    # Labels and title
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Per-Image Pixel Error (px)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Pose Estimation Error Distribution - {dataset_name} Dataset',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=8, label='Mean'),
        Line2D([0], [0], color='orange', linewidth=2, label='Median'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved error distribution box plot to: {output_path}")


def plot_timing_comparison(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Create a bar chart comparing inference times across models.
    
    Args:
        df: DataFrame with per-image metrics including 'inference_time_ms' and 'model_name'
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    # Check if timing column exists
    if "inference_time_ms" not in df.columns:
        print("\n⚠ Timing data not found. Skipping timing visualization.")
        return
    
    # Filter out NaN timing
    df_clean = df.dropna(subset=["inference_time_ms"])
    
    if df_clean.empty:
        print("\n⚠ No valid timing data found. Skipping timing visualization.")
        return
    
    # Compute mean and std per model
    timing_stats = df_clean.groupby("model_name")["inference_time_ms"].agg(["mean", "std"]).reset_index()
    timing_stats = timing_stats.sort_values("mean")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    x_pos = np.arange(len(timing_stats))
    bars = ax.bar(
        x_pos,
        timing_stats["mean"],
        yerr=timing_stats["std"],
        capsize=5,
        alpha=0.7,
        color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'][:len(timing_stats)],
        edgecolor='black',
    )
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(
        zip(bars, timing_stats["mean"], timing_stats["std"])
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std_val + 1,
            f'{mean_val:.1f}ms\n±{std_val:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )
    
    # Labels and title
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Model Inference Time Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(timing_stats["model_name"])
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved timing comparison to: {output_path}")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize pose estimation benchmark results"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        required=True,
        help="Path to directory containing per-image metrics CSV files (e.g., outputs/metrics/mpii)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save generated figures (default: outputs/figures)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MPII",
        help="Dataset name for plot titles (default: MPII)",
    )
    
    args = parser.parse_args()
    
    # Convert paths
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading metrics from: {metrics_dir}")
    
    try:
        # Load per-image metrics
        df = load_per_image_metrics(metrics_dir)
        print(f"✓ Loaded {len(df)} records for {df['model_name'].nunique()} models")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate error distribution box plot
        try:
            error_plot_path = output_dir / "error_distribution_boxplot.png"
            plot_error_distribution_boxplot(
                df,
                error_plot_path,
                dataset_name=args.dataset_name,
            )
        except ValueError as e:
            print(f"\n⚠ Skipping error distribution plot: {e}")
        
        # Generate timing comparison
        timing_plot_path = output_dir / "timing_comparison.png"
        plot_timing_comparison(df, timing_plot_path)
        
        print(f"\n✓ All visualizations complete! Figures saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
