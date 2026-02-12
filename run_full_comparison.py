#!/usr/bin/env python3
"""
Complete comparative benchmarking pipeline.
Runs benchmarks, analysis, and visualization in one command.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}\n")


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command list to execute
        description: Description of what's being run
        
    Returns:
        True if successful, False otherwise
    """
    print(f"▶ {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}\n")
        return False
    except Exception as e:
        print(f"✗ Error during {description}: {e}\n")
        return False


def display_comparison_summary(comparison_path: Path) -> None:
    """Display comparison summary from CSV."""
    if not comparison_path.exists():
        print("⚠ Comparison CSV not found")
        return
    
    df = pd.read_csv(comparison_path)
    
    print_section("📊 COMPARISON SUMMARY")
    
    # Key metrics table
    print("┌" + "─" * 78 + "┐")
    print("│ " + "Model".ljust(15) + " │ " + 
          "MPII Err".rjust(10) + " │ " +
          "Workout Err".rjust(12) + " │ " +
          "Δ Error".rjust(10) + " │ " +
          "Δ %".rjust(8) + " │ " +
          "Conf Drop".rjust(10) + " │")
    print("├" + "─" * 78 + "┤")
    
    for _, row in df.iterrows():
        symbol = "✅" if row['error_pct_increase'] < 20 else "⚠️" if row['error_pct_increase'] < 50 else "❌"
        print(f"│ {symbol} {row['model_name']:12s} │ " +
              f"{row['mean_pixel_error_mpii']:9.2f}px │ " +
              f"{row['mean_pixel_error_workout']:11.2f}px │ " +
              f"{row['error_delta_px']:9.2f}px │ " +
              f"{row['error_pct_increase']:7.1f}% │ " +
              f"{row['conf_drop']:9.3f} │")
    
    print("└" + "─" * 78 + "┘")
    
    # Key findings
    best = df.loc[df['error_pct_increase'].idxmin()]
    worst = df.loc[df['error_pct_increase'].idxmax()]
    
    print(f"\n🏆 Most Robust Model: {best['model_name']}")
    print(f"   Error increase: {best['error_delta_px']:.2f}px ({best['error_pct_increase']:.1f}%)")
    
    print(f"\n⚠️  Most Degraded Model: {worst['model_name']}")
    print(f"   Error increase: {worst['error_delta_px']:.2f}px ({worst['error_pct_increase']:.1f}%)")
    
    # Overall stats
    avg_increase = df['error_pct_increase'].mean()
    print(f"\n📈 Average error increase across all models: {avg_increase:.1f}%")


def display_outputs(output_dir: Path) -> None:
    """Display generated output files."""
    print_section("📁 GENERATED OUTPUTS")
    
    outputs = [
        ("MPII leaderboard", output_dir / "leaderboard_mpii.csv"),
        ("Workout leaderboard", output_dir / "leaderboard_workout.csv"),
        ("Comparison CSV", output_dir / "leaderboard_comparison.csv"),
        ("Comparison summary", output_dir / "comparison" / "comparison_summary.md"),
        ("Side-by-side box plot", output_dir / "comparison" / "figures" / "error_comparison_boxplot.png"),
        ("Error delta chart", output_dir / "comparison" / "figures" / "error_delta_barchart.png"),
        ("Confidence scatter", output_dir / "comparison" / "figures" / "confidence_comparison_scatter.png"),
    ]
    
    for name, path in outputs:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024  # KB
                if size > 1024:
                    size_str = f"{size/1024:.1f} MB"
                else:
                    size_str = f"{size:.1f} KB"
                print(f"  ✓ {name}: {path}")
                print(f"    └─ {size_str}")
            else:
                print(f"  ✓ {name}: {path}")
        else:
            print(f"  ⚠ {name}: Not found")
    
    print()


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    print_banner("🔬 COMPARATIVE POSE ESTIMATION PIPELINE", "═")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    output_dir = Path("outputs")
    
    # Step 1: Run comparative benchmark
    print_section("🔬 STEP 1/4: Running Comparative Benchmark")
    print("Datasets: MPII + Workout")
    print("This may take several minutes...\n")
    
    success = run_command(
        ["uv", "run", "python", "run_comparative_benchmark.py", "--dataset", "both"],
        "Comparative benchmark execution"
    )
    
    if not success:
        print_banner("❌ BENCHMARK FAILED", "═")
        print("Check that:")
        print("  - config_comparative.yaml is properly configured")
        print("  - Dataset directories exist and contain images")
        print("  - MPII annotations are available")
        sys.exit(1)
    
    # Step 2: Compare datasets
    print_section("📊 STEP 2/4: Analyzing Comparison Metrics")
    
    success = run_command(
        ["uv", "run", "python", "compare_datasets.py"],
        "Dataset comparison analysis"
    )
    
    if not success:
        print("⚠ Comparison analysis failed, but benchmark results are available")
    
    # Step 3: Generate visualizations
    print_section("📈 STEP 3/4: Generating Visualizations")
    
    success = run_command(
        ["uv", "run", "python", "visualize_comparison.py"],
        "Comparative visualization generation"
    )
    
    if not success:
        print("⚠ Visualization failed, but analysis results are available")
    
    # Step 4: Display results
    print_section("🎯 STEP 4/4: Results Summary")
    
    # Display comparison summary
    comparison_path = output_dir / "leaderboard_comparison.csv"
    if comparison_path.exists():
        display_comparison_summary(comparison_path)
    else:
        print("⚠ Comparison summary not available")
    
    # Display outputs
    display_outputs(output_dir)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    minutes = int(duration.total_seconds() // 60)
    seconds = int(duration.total_seconds() % 60)
    
    print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY", "═")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Quick access commands
    print("Quick access:")
    print(f"  View comparison:  cat {output_dir}/comparison/comparison_summary.md")
    print(f"  View figures:     open {output_dir}/comparison/figures")
    print(f"  View leaderboards: cat {output_dir}/leaderboard_comparison.csv")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
