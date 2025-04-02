# examples/run_comparative_benchmark.py
"""Benchmarking and comparison of multiple molecular subgraph isomorphism algorithms."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from prism.benchmark.adapters.algorithm_adapters import get_all_algorithm_adapters
from prism.benchmark.configs.benchmark_config import BenchmarkConfig
from prism.benchmark.services.comparative_benchmark import ComparativeBenchmarkRunner
from prism.benchmark.services.comparative_visualizer import ComparativeVisualizer
from prism.benchmark.services.dataset_generator import DatasetGenerator


def main():
    """Run a comparative benchmark of multiple molecular subgraph isomorphism algorithms."""
    print("=== PRISM Comparative Benchmark ===")

    parser = argparse.ArgumentParser(
        description="Run comparative benchmarks of molecular graph matching algorithms"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to the benchmark dataset JSON file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes to use (1-12)",
    )
    parser.add_argument(
        "--output_dir",
        default="benchmark_results",
        help="Directory to store benchmark results",
    )

    args = parser.parse_args()

    # Validate input dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print(
            "Please generate a dataset first using 'examples/generate_benchmark_dataset.py'"
        )
        sys.exit(1)

    # Validate number of workers
    num_workers = max(1, min(12, args.workers))

    # Create output directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)

    print(f"Using dataset: {dataset_path}")

    # Load the dataset
    try:
        generator = DatasetGenerator(
            BenchmarkConfig(
                name="Comparative Benchmark",
                description="Comparison of molecular subgraph isomorphism algorithms",
                pairs_per_category=20,
            )
        )
        generator.load_molecular_dataset(str(dataset_path))
        dataset = generator.load_benchmark_dataset(str(dataset_path))
        print(f"Loaded dataset with {len(dataset.pairs)} test cases")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Configure algorithm parameters
    algorithm_params = {
        "prism_params": {
            "signature_params": {
                "max_distance": 2,
                "use_bond_types": True,
                "use_cycles": True,
            },
            "compatibility_params": {
                "element_match_required": True,
                "min_signature_similarity": 0.7,
                "bond_type_match_required": True,
                "progressive_refinement": True,
                "max_compatible_degree_diff": 1,
            },
            "match_params": {
                "max_time_seconds": 30,
                "max_iterations": 1000000,
                "num_threads": 4,
                "use_forward_checking": True,
                "use_conflict_backjumping": True,
            },
        },
        "rdkit_params": {
            "timeout": 30,
            "completeRingsOnly": False,
            "ringMatchesRingOnly": False,
            "matchValences": True,
            "atomCompare": "elements",
            "bondCompare": "order",
        },
        "mcgregor_params": {"timeout": 30, "match_rings": False, "match_bonds": True},
        "vf2_params": {
            "timeout": 30,
            "use_node_attributes": True,
            "use_edge_attributes": True,
            "match_rings": False,
        },
    }

    # Get all available algorithm adapters
    try:
        algorithms = get_all_algorithm_adapters(**algorithm_params)
        if not algorithms:
            print("Error: No algorithm adapters found")
            sys.exit(1)

        print(f"Found {len(algorithms)} available algorithms:")
        for alg_name in algorithms.keys():
            print(f"  - {alg_name}")

        # Check for expected algorithms
        expected_algorithms = ["PRISM", "RDKit-MCS", "VF2", "McGregor"]
        missing_algorithms = [
            alg for alg in expected_algorithms if alg not in algorithms
        ]

        if missing_algorithms:
            print(
                f"Warning: Some expected algorithms are missing: {', '.join(missing_algorithms)}"
            )
            print("This may affect the completeness of your benchmark comparison")
    except Exception as e:
        print(f"Error loading algorithm adapters: {e}")
        sys.exit(1)

    print(f"Using {num_workers} workers for parallel execution")

    # Create the benchmark runner
    runner = ComparativeBenchmarkRunner(
        datasets=dataset,
        algorithms=algorithms,
        algorithm_params=algorithm_params,
        num_workers=num_workers,
        use_processes=True,  # Use processes for parallel execution
    )

    # Run the comparative benchmark
    print("\nRunning comparative benchmark...")
    start_time = time.time()
    result = runner.run_comparative_benchmark(
        name="Algorithm Comparison Benchmark",
        description="Comparison of molecular subgraph isomorphism algorithms across diverse molecular structures",
    )
    end_time = time.time()

    # Save the results
    results_path = (
        results_dir / f"comparative_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result.save_results(str(results_path))

    # Print summary
    print(f"\nBenchmark completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {results_path}")

    # Convert to dataframe for display
    df = result.to_dataframe()

    # Show key metrics in a summary table
    print("\n=== Algorithm Comparison ===")
    metrics_to_display = [
        "algorithm",
        "avg_match_size",
        "avg_match_ratio",
        "avg_time",
        "tests_completed",
    ]

    # Add quality metrics if available (for algorithms with reference solutions)
    if any(
        column in df.columns for column in ["avg_precision", "avg_recall", "avg_f1"]
    ):
        quality_metrics = [
            col
            for col in ["avg_precision", "avg_recall", "avg_f1"]
            if col in df.columns
        ]
        metrics_to_display.extend(quality_metrics)

    print(df[metrics_to_display].to_string(index=False))

    # Create visualizations
    print("\nGenerating visualizations...")
    visualizer = ComparativeVisualizer(result)

    # Create plots directory
    plots_dir = results_dir / "comparative_plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate and save all plots
    visualizer.save_plots(output_dir=str(plots_dir))

    print(f"\nVisualizations saved to {plots_dir}")


if __name__ == "__main__":
    main()
