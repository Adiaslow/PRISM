# examples/run_benchmark.py
"""Example script for running PRISM benchmarks."""

import os
import sys
import time
import json
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from prism.benchmark.core.implementations.benchmark import BenchmarkRunner
from prism.benchmark.services.dataset_generator import DatasetGenerator
from prism.benchmark.configs.benchmark_config import BenchmarkConfig
from prism.algorithm.graph_matcher import MatchParameters, GraphMatcher
from prism.benchmark.services.visualizer import BenchmarkVisualizer
from prism.core.compatibility_matrix import CompatibilityParameters
from prism.algorithm.seed_selection import SeedParameters, SeedPriority, SeedSelector


def load_config(config_name):
    """Load algorithm configuration from config file.

    Args:
        config_name: Name of the configuration to load

    Returns:
        dict: Dictionary containing the configuration parameters
    """
    config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.json"

    if not config_path.exists():
        print(f"Error: Configuration '{config_name}' not found at {config_path}")
        print("Available configurations:")
        for config_file in (Path(__file__).parent.parent / "configs").glob("*.json"):
            print(f"  - {config_file.stem}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded configuration: {config['name']}")
    print(f"Description: {config['description']}")

    return config


def setup_algorithm_with_config(config):
    """Set up the algorithm with the given configuration.

    Args:
        config: Dictionary containing the configuration parameters

    Returns:
        tuple: (signature_params, compatibility_params, match_params)
    """
    # Extract parameters from config
    signature_params = config["signature_params"]

    # Create objects for the parameters that need them
    compatibility_params = CompatibilityParameters(**config["compatibility_params"])
    match_params = MatchParameters(**config["match_params"])

    # Set up seed selection if needed
    if "seed_params" in config:
        seed_config = config["seed_params"]
        seed_priority = SeedPriority(**seed_config["priority"])
        seed_params = SeedParameters(
            max_seeds=seed_config["max_seeds"],
            diversity_threshold=seed_config["diversity_threshold"],
            min_compatibility_score=seed_config["min_compatibility_score"],
            use_weighted_sampling=seed_config["use_weighted_sampling"],
            priority=seed_priority,
        )

        # Monkey patch the GraphMatcher to use our seed parameters
        original_find_mcs = GraphMatcher.find_maximum_common_subgraph

        def patched_find_mcs(self, seeds=None):
            """Patched version to use configured seed selection parameters"""
            # Create custom seed selector with configured parameters if no seeds provided
            if seeds is None:
                seed_selector = SeedSelector(
                    self.graph_a, self.graph_b, self.compatibility_matrix, seed_params
                )
                seeds = seed_selector.select_seeds()

            # Call the original method with selected seeds
            return original_find_mcs(self, seeds)

        # Apply the monkey patch
        GraphMatcher.find_maximum_common_subgraph = patched_find_mcs

    return signature_params, compatibility_params, match_params


def main():
    """Run a benchmark of the PRISM algorithm."""
    print("=== PRISM Benchmark ===")

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run PRISM benchmark")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--dataset", type=str, help="Path to existing dataset file")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration to use (default, diverse_atoms, hyperopt)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories for results
    results_dir = Path(__file__).parent.parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    # Use existing dataset or generate a new one
    if args.dataset:
        print(f"Using existing dataset: {args.dataset}")
        dataset_path = args.dataset
        # Load the dataset using the class method
        dataset = DatasetGenerator.load_dataset(args.dataset)
        print(f"Loaded dataset with {len(dataset.pairs)} pairs")
    else:
        # Generate dataset
        print("Generating benchmark dataset...")

        # Create config with only specific categories for the demo
        categories = {
            "SB-Small": BenchmarkConfig.default_config().categories["SB-Small"],
            "SB-Medium": BenchmarkConfig.default_config().categories["SB-Medium"],
            "SI-Small/Medium": BenchmarkConfig.default_config().categories[
                "SI-Small/Medium"
            ],
        }

        config = BenchmarkConfig(
            name="Demo Benchmark Dataset",
            description="Small demonstration dataset for PRISM benchmark",
            categories=categories,
            pairs_per_category=3,  # Small number for demonstration
        )

        generator = DatasetGenerator(config)

        # Generate the dataset
        dataset = generator.generate_dataset()

        # Save the dataset (using the generator's save_dataset method)
        dataset_path = results_dir / "demo_dataset.json"
        generator.dataset = dataset
        generator.save_dataset(str(dataset_path))
        print(f"Dataset saved to {dataset_path}")

    # Configure the benchmark runner using loaded configuration
    signature_params, compatibility_params, match_params = setup_algorithm_with_config(
        config
    )

    # Create the benchmark runner
    runner = BenchmarkRunner(
        datasets=dataset,
        signature_params=signature_params,
        compatibility_params=compatibility_params.__dict__,  # Convert to dict for the runner
        match_params=match_params.__dict__,  # Convert to dict for the runner
        num_workers=4,  # Use 4 workers for parallel execution
        use_processes=True,  # Use processes for parallel execution
    )

    # Run the benchmark
    print("Running benchmark...")
    start_time = time.time()
    result = runner.run_benchmark(name=f"PRISM Benchmark - {config['name']}")
    end_time = time.time()

    # Save the results
    results_path = (
        results_dir / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result.save_results(str(results_path))

    # Print summary
    print(f"\nBenchmark completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {results_path}")

    # Display summary metrics
    if result.metrics:
        print("\n=== Benchmark Summary ===")
        print(f"Number of tests: {len(result.test_results)}")
        print(f"Overall accuracy: {result.metrics.accuracy:.4f}")

        # Handle metrics with safer access
        try:
            # F1 score - use avg_f1
            if hasattr(result.metrics, "avg_f1"):
                f1_score = result.metrics.avg_f1
            else:
                f1_score = 0.0
            print(f"Average F1 score: {f1_score:.4f}")

            # Execution time - use avg_time
            if hasattr(result.metrics, "avg_time"):
                exec_time = result.metrics.avg_time
            else:
                exec_time = 0.0
            print(f"Average execution time: {exec_time:.4f} seconds")

            # Print category breakdown if by_category exists
            if hasattr(result.metrics, "by_category"):
                print("\n=== Performance by Category ===")
                for category, metrics in result.metrics.by_category.items():
                    print(f"{category}:")
                    print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                    print(f"  F1 Score: {metrics.get('f1', 0.0):.4f}")
                    print(f"  Avg Time: {metrics.get('avg_time', 0.0):.4f} seconds")
        except Exception as e:
            print(f"Error displaying metrics: {e}")

    # Visualize results
    if not args.no_vis:
        print("\n=== Generating Visualizations ===")
        # Create plots directory
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Create visualizer
        visualizer = BenchmarkVisualizer(result)

        # Generate visualizations
        print("Generating time by size plot...")
        time_size_plot = visualizer.plot_time_by_size(
            save_path=str(plots_dir / "time_by_size.png")
        )

        print("Generating time by category plot...")
        time_category_plot = visualizer.plot_time_by_category(
            save_path=str(plots_dir / "time_by_category.png")
        )

        print("Generating metrics heatmap...")
        metrics_heatmap = visualizer.plot_metrics_heatmap(
            save_path=str(plots_dir / "metrics_heatmap.png")
        )

        print("Generating size comparison plot...")
        size_comparison = visualizer.plot_size_comparison(
            save_path=str(plots_dir / "size_comparison.png")
        )

        # Generate summary report
        print("Generating summary report...")
        report_text = visualizer.generate_summary_report(
            save_path=str(plots_dir / "summary_report.txt")
        )

        print(f"\nVisualizations saved to {plots_dir}")
        print("Generated files:")
        for file in plots_dir.glob("*"):
            print(f"  - {file.name}")


if __name__ == "__main__":
    main()
