#!/usr/bin/env python3
"""
Run complete pose estimation benchmark pipeline.
Executes benchmarking, visualization, and presents results.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


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


def format_table_row(values: list, widths: list) -> str:
    """Format a table row with aligned columns."""
    return " │ ".join(
        str(val).ljust(width) if i == 0 else str(val).rjust(width)
        for i, (val, width) in enumerate(zip(values, widths))
    )


def display_leaderboard(leaderboard_path: Path) -> None:
    """Display leaderboard in a pretty format."""
    if not leaderboard_path.exists():
        print("⚠ Leaderboard file not found")
        return
    
    df = pd.read_csv(leaderboard_path)
    
    # Define column widths and display names
    col_display = {
        "model_name": ("Model", 15),
        "num_images": ("Images", 7),
        "pose_rate": ("Pose Rate", 10),
        "mean_detected_joints": ("Avg Joints", 10),
        "mean_conf": ("Confidence", 10),
        "mean_pixel_error": ("Error (px)", 10),
        "mean_inference_ms": ("Infer (ms)", 11),
        "std_inference_ms": ("Std (ms)", 9),
    }
    
    # Get columns that exist in the dataframe
    display_cols = [col for col in col_display.keys() if col in df.columns]
    headers = [col_display[col][0] for col in display_cols]
    widths = [col_display[col][1] for col in display_cols]
    
    # Print table
    print_section("📊 BENCHMARK LEADERBOARD")
    
    # Header
    print("┌" + "─┬─".join("─" * w for w in widths) + "┐")
    print("│ " + format_table_row(headers, widths) + " │")
    print("├" + "─┼─".join("─" * w for w in widths) + "┤")
    
    # Rows
    for _, row in df.iterrows():
        values = []
        for col in display_cols:
            val = row[col]
            # Format numbers nicely
            if col == "pose_rate":
                values.append(f"{val:.1f}%")
            elif col in ["mean_detected_joints", "mean_pixel_error"]:
                values.append(f"{val:.2f}")
            elif col in ["mean_conf"]:
                values.append(f"{val:.3f}")
            elif col in ["mean_inference_ms", "std_inference_ms"]:
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        print("│ " + format_table_row(values, widths) + " │")
    
    print("└" + "─┴─".join("─" * w for w in widths) + "┘")


def display_summary(outputs_dir: Path) -> None:
    """Display summary of generated outputs."""
    print_section("📁 GENERATED OUTPUTS")
    
    outputs = [
        ("Leaderboard CSV", outputs_dir / "leaderboard.csv"),
        ("Per-image metrics", outputs_dir / "metrics"),
        ("Overlay images", outputs_dir / "overlays"),
        ("Error distribution plot", outputs_dir / "figures" / "error_distribution_boxplot.png"),
        ("Timing comparison plot", outputs_dir / "figures" / "timing_comparison.png"),
    ]
    
    for name, path in outputs:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024  # KB
                print(f"  ✓ {name}: {path}")
                if size > 1024:
                    print(f"    └─ Size: {size/1024:.1f} MB")
                else:
                    print(f"    └─ Size: {size:.1f} KB")
            else:
                # Count files in directory
                files = list(path.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                print(f"  ✓ {name}: {path}")
                print(f"    └─ {file_count} files")
        else:
            print(f"  ⚠ {name}: Not found")
    
    print()


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    print_banner("🏃 POSE ESTIMATION BENCHMARK PIPELINE", "═")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config_path = Path("config.yaml")
    outputs_dir = Path("outputs")
    max_images = 150  # Default, can be changed
    
    if len(sys.argv) > 1:
        try:
            max_images = int(sys.argv[1])
            print(f"Using {max_images} images from command line argument")
        except ValueError:
            print(f"⚠ Invalid max_images argument: {sys.argv[1]}, using default: {max_images}")
    
    # Step 1: Run benchmark
    print_section("🔬 STEP 1/3: Running Benchmark")
    print(f"Config: {config_path}")
    print(f"Max images: {max_images}")
    print(f"Output directory: {outputs_dir}\n")
    
    success = run_command(
        ["uv", "run", "python", "-m", "pose_bench.run_benchmark",
         "--config", str(config_path),
         "--max-images", str(max_images)],
        "Benchmark execution"
    )
    
    if not success:
        print_banner("❌ BENCHMARK FAILED", "═")
        sys.exit(1)
    
    # Step 2: Generate visualizations
    print_section("📊 STEP 2/3: Generating Visualizations")
    
    success = run_command(
        ["uv", "run", "python", "-m", "pose_bench.visualize_results",
         "--metrics-dir", "outputs/metrics/mpii",
         "--output-dir", "outputs/figures",
         "--dataset-name", "MPII"],
        "Visualization generation"
    )
    
    if not success:
        print("⚠ Visualization failed, but benchmark results are available")
    
    # Step 3: Display results
    print_section("🎯 STEP 3/3: Results Summary")
    
    # Display leaderboard
    leaderboard_path = outputs_dir / "leaderboard.csv"
    display_leaderboard(leaderboard_path)
    
    # Display output summary
    display_summary(outputs_dir)
    
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
    print(f"  View leaderboard: cat {leaderboard_path}")
    print(f"  View figures:     open {outputs_dir / 'figures'}")
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
