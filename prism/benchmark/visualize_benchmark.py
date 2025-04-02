# examples/visualize_benchmark.py
"""Example script for visualizing PRISM benchmark results."""

import glob
import os
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from prism.benchmark.core.implementations.benchmark import BenchmarkResult
from prism.benchmark.services.visualizer import BenchmarkVisualizer


def main():
    """Visualize benchmark results."""
    print("=== PRISM Benchmark Visualizer ===")

    # Check for results
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found.")
        print("Please run examples/run_benchmark.py first to generate results.")
        return

    # Find the most recent result file
    result_files = list(results_dir.glob("benchmark_results_*.json"))
    if not result_files:
        print(f"Error: No benchmark result files found in '{results_dir}'.")
        print("Please run examples/run_benchmark.py first to generate results.")
        return

    # Sort by modification time
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent = result_files[0]

    print(f"Found {len(result_files)} result files.")
    print(f"Using most recent: {most_recent}")

    # Load the benchmark result
    result = BenchmarkResult.load_results(most_recent)

    # Create visualizer
    visualizer = BenchmarkVisualizer(result)

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate and save all plots
    print(f"Generating plots in '{plots_dir}'...")
    visualizer.save_plots(output_dir=str(plots_dir))

    print(f"\nVisualization complete! Plots saved to {plots_dir}")
    print("\nIndividual plots generated:")
    for plot_file in plots_dir.glob("*.png"):
        print(f"- {plot_file.name}")

    # Show summary report
    report_path = plots_dir / "summary_report.txt"
    print(f"\nSummary report generated at {report_path}")

    # Print a snippet of the report
    if report_path.exists():
        with open(report_path, "r") as f:
            lines = f.readlines()
            # Print the first 15 lines of the report
            print("\n=== Report Preview ===")
            print("".join(lines[:15]))
            print("...\n")

    print(
        "To view the complete report and plots, open the files in the plots directory."
    )


if __name__ == "__main__":
    main()
