# examples/hyperparameter_tuning.py
"""Hyperparameter optimization for the PRISM algorithm."""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import traceback
from typing import Dict, Any

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import PRISM modules
import prism
from prism.benchmark.services.benchmark import BenchmarkRunner
from prism.benchmark.services.dataset_generator import DatasetGenerator, BenchmarkConfig
from prism.algorithm.graph_matcher import MatchParameters, GraphMatcher
from prism.core.compatibility_matrix import CompatibilityParameters
from prism.algorithm.seed_selection import SeedParameters, SeedPriority, SeedSelector

# Import Optuna (needs to be installed via pip)
try:
    import optuna
    from optuna.visualization import plot_param_importances, plot_optimization_history

    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: Optuna is not installed. Install using 'pip install optuna plotly'")
    OPTUNA_AVAILABLE = False

# Global variable to store dataset
dataset = None


def run_benchmark_with_params(params):
    """Run benchmark with the given parameters."""
    try:
        # Create benchmark runner
        runner = BenchmarkRunner(
            datasets=sample_dataset(dataset, max_pairs=50),
            signature_params=params["signature"],
            compatibility_params=params["compatibility"],
            match_params=params["match"],
            num_workers=params["performance"]["num_threads"],
            use_processes=True,
        )

        # Run benchmark
        result = runner.run_benchmark(name=f"Speed Optimization")
        return result

    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        traceback.print_exc()
        return None


def objective(trial, dataset_path):
    """Objective function for hyperparameter optimization."""
    global dataset  # Use global dataset to avoid reloading

    # Load dataset if not already loaded
    if dataset is None:
        generator = DatasetGenerator(
            BenchmarkConfig(
                name="Hyperparameter Optimization",
                description="Dataset for hyperparameter optimization",
                pairs_per_category=20,
            )
        )
        generator.load_molecular_dataset(dataset_path)
        dataset = generator.load_benchmark_dataset(dataset_path)

    # Configure parameters
    params = {
        "signature": {
            "max_distance": trial.suggest_int("signature_max_distance", 2, 5),
        },
        "compatibility": {
            "progressive_refinement": trial.suggest_categorical(
                "comp_progressive_refinement", [True, False]
            ),
            "element_match_required": True,
            "bond_type_match_required": True,
            "min_signature_similarity": 0.9,
            "max_compatible_degree_diff": 0,
            "use_sparse_matrix": True,
        },
        "seed": {
            "signature_uniqueness_weight": trial.suggest_float(
                "seed_signature_uniqueness_weight", 0.1, 0.9
            ),
            "structural_importance_weight": trial.suggest_float(
                "seed_structural_importance_weight", 0.1, 0.9
            ),
            "expansion_potential_weight": trial.suggest_float(
                "seed_expansion_potential_weight", 0.1, 0.9
            ),
            "connectivity_weight": trial.suggest_float(
                "seed_connectivity_weight", 0.1, 0.9
            ),
            "max_seeds": trial.suggest_int(
                "seed_max_seeds", 3, 20
            ),  # Reduced from 50 to focus on speed
            "diversity_threshold": trial.suggest_float(
                "seed_diversity_threshold", 0.3, 0.9
            ),
            "min_compatibility_score": trial.suggest_float(
                "seed_min_compatibility_score", 0.7, 0.95
            ),
            "use_weighted_sampling": trial.suggest_categorical(
                "seed_use_weighted_sampling", [True, False]
            ),
        },
        "match": {
            "max_iterations": trial.suggest_int(
                "match_max_iterations", 100000, 500000
            ),  # Reduced max iterations
            "max_time_seconds": 60,
            "use_forward_checking": True,
            "use_conflict_backjumping": True,
        },
        "performance": {
            "num_threads": trial.suggest_int("num_threads", 1, 12),
        },
    }

    # Run benchmark with these parameters
    result = run_benchmark_with_params(params)
    if result is None or result.metrics is None:
        print(f"Trial failed - returning penalty score")
        return 2000.0  # Penalty for failed trials

    avg_match_size = getattr(result.metrics, "avg_match_size", 0)
    avg_match_ratio = getattr(result.metrics, "avg_match_ratio", 0)
    avg_time = getattr(result.metrics, "avg_time", float("inf"))

    # Print trial metrics for monitoring
    print(f"  Average Match Size: {avg_match_size:.2f}")
    print(f"  Average Match Ratio: {avg_match_ratio:.2f}")
    print(f"  Average Time: {avg_time:.6f} seconds")

    # Check if matches are exact (within small epsilon)
    EXPECTED_MATCH_SIZE = 1.13
    EXPECTED_MATCH_RATIO = 0.15
    EPSILON = 0.01

    if (
        abs(avg_match_size - EXPECTED_MATCH_SIZE) > EPSILON
        or abs(avg_match_ratio - EXPECTED_MATCH_RATIO) > EPSILON
    ):
        penalty = (
            1000.0
            + abs(avg_match_size - EXPECTED_MATCH_SIZE)
            + abs(avg_match_ratio - EXPECTED_MATCH_RATIO)
        )
        print(f"  Score: {penalty:.2f} (Match size differs from expected)")
        return penalty

    # If matches are exact, optimize for speed
    score = avg_time * 1000  # Convert to milliseconds
    print(f"  Score: {score:.6f} (Optimizing speed)")
    return score


def sample_dataset(dataset, max_pairs=50):
    """Sample a smaller subset of the dataset for faster evaluation.

    Args:
        dataset: Full benchmark dataset
        max_pairs: Maximum number of pairs to include

    Returns:
        BenchmarkDataset: Sampled dataset
    """
    from prism.core.molecular_pair import BenchmarkDataset

    # Get available categories
    categories = {}
    for pair in dataset.pairs:
        if pair.category not in categories:
            categories[pair.category] = []
        categories[pair.category].append(pair)

    # Calculate pairs per category for balanced sampling
    pairs_per_category = max(1, min(5, max_pairs // len(categories)))

    # Sample pairs
    sampled_pairs = []
    for category, pairs in categories.items():
        # Take random sample from each category
        if len(pairs) > pairs_per_category:
            import random

            sampled = random.sample(pairs, pairs_per_category)
        else:
            sampled = pairs
        sampled_pairs.extend(sampled)

    # Create new dataset
    sampled_dataset = BenchmarkDataset(
        name=f"Sampled {dataset.name}",
        description=f"Sampled subset of {dataset.name} for optimization",
        pairs=sampled_pairs,
    )

    return sampled_dataset


def save_config_file(
    best_params: Dict[str, Any],
    best_metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """Save the best hyperparameters and metrics to a configuration file."""
    config = {
        "signature": {
            "max_distance": best_params["signature_max_distance"],
        },
        "compatibility": {
            "progressive_refinement": best_params["comp_progressive_refinement"],
        },
        "seed": {
            "signature_uniqueness_weight": best_params[
                "seed_signature_uniqueness_weight"
            ],
            "structural_importance_weight": best_params[
                "seed_structural_importance_weight"
            ],
            "expansion_potential_weight": best_params[
                "seed_expansion_potential_weight"
            ],
            "connectivity_weight": best_params["seed_connectivity_weight"],
            "max_seeds": best_params["seed_max_seeds"],
            "diversity_threshold": best_params["seed_diversity_threshold"],
            "min_compatibility_score": best_params["seed_min_compatibility_score"],
            "use_weighted_sampling": best_params["seed_use_weighted_sampling"],
        },
        "match": {
            "max_iterations": best_params["match_max_iterations"],
        },
        "performance": {
            "num_threads": best_params["num_threads"],
        },
        "metrics": {
            "avg_match_size": best_metrics["avg_match_size"],
            "avg_match_ratio": best_metrics["avg_match_ratio"],
            "avg_time": best_metrics["avg_time"],
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved best configuration to: {config_path}")


def run_optimization(dataset_path, n_trials=100, study_name="prism_hyperopt"):
    """Run hyperparameter optimization.

    Args:
        dataset_path: Path to benchmark dataset
        n_trials: Number of optimization trials to run
        study_name: Name for the optimization study

    Returns:
        dict: Best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. Install with 'pip install optuna plotly'"
        )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize our composite score
        sampler=optuna.samplers.TPESampler(
            seed=42
        ),  # Use TPE algorithm with fixed seed
    )

    # Run optimization
    objective_func = partial(objective, dataset_path=dataset_path)
    study.optimize(objective_func, n_trials=n_trials)

    # Print results
    print(f"\n=== Optimization Results ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.4f}")
    print(f"Best avg_match_size: {study.best_trial.user_attrs['avg_match_size']:.4f}")
    print(f"Best avg_match_ratio: {study.best_trial.user_attrs['avg_match_ratio']:.4f}")
    print(f"Best avg_time: {study.best_trial.user_attrs['avg_time']:.6f} seconds")

    print("\nBest hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")

    # Save study results
    results_dir = Path(__file__).parent.parent / "optimization_results"
    results_dir.mkdir(exist_ok=True)

    # Save best parameters as JSON
    params_file = results_dir / f"best_params_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(params_file, "w") as f:
        json.dump(
            {
                "best_params": study.best_params,
                "best_avg_match_size": study.best_trial.user_attrs["avg_match_size"],
                "best_avg_match_ratio": study.best_trial.user_attrs["avg_match_ratio"],
                "best_avg_time": study.best_trial.user_attrs["avg_time"],
                "best_score": study.best_trial.value,
            },
            f,
            indent=2,
        )

    # Save in standard config format
    save_config_file(
        study.best_params,
        {
            "avg_match_size": study.best_trial.user_attrs["avg_match_size"],
            "avg_match_ratio": study.best_trial.user_attrs["avg_match_ratio"],
            "avg_time": study.best_trial.user_attrs["avg_time"],
        },
        str(results_dir),
    )

    # Generate and save visualization
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.write_image(str(results_dir / "optimization_history.png"))
        fig2.write_image(str(results_dir / "param_importances.png"))
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        print("Make sure plotly is installed: pip install plotly kaleido")

    # Return best parameters
    return study.best_params


def validate_best_params(dataset_path, best_params):
    """Validate the best parameters on the full dataset.

    Args:
        dataset_path: Path to benchmark dataset
        best_params: Dictionary of best parameters from optimization

    Returns:
        dict: Validation results
    """
    print("\n=== Validating Best Parameters ===")

    # Load dataset
    dataset = DatasetGenerator.load_dataset(dataset_path)

    # Configure parameters
    signature_params = {
        "max_distance": best_params["signature_max_distance"],
        "use_bond_types": best_params["signature_use_bond_types"],
        "use_cycles": best_params["signature_use_cycles"],
    }

    compatibility_params = CompatibilityParameters(
        element_match_required=best_params["comp_element_match_required"],
        min_signature_similarity=best_params["comp_min_signature_similarity"],
        bond_type_match_required=best_params["comp_bond_type_match_required"],
        progressive_refinement=best_params["comp_progressive_refinement"],
        max_compatible_degree_diff=best_params["comp_max_compatible_degree_diff"],
        use_sparse_matrix=True,
    )

    seed_priority = SeedPriority(
        signature_uniqueness_weight=best_params["seed_signature_uniqueness_weight"],
        structural_importance_weight=best_params["seed_structural_importance_weight"],
        expansion_potential_weight=best_params["seed_expansion_potential_weight"],
        connectivity_weight=best_params["seed_connectivity_weight"],
    )

    seed_params = SeedParameters(
        max_seeds=best_params["seed_max_seeds"],
        diversity_threshold=best_params["seed_diversity_threshold"],
        min_compatibility_score=best_params["seed_min_compatibility_score"],
        use_weighted_sampling=best_params["seed_use_weighted_sampling"],
        priority=seed_priority,
    )

    match_params = MatchParameters(
        max_time_seconds=60,  # Use longer timeout for final validation
        max_iterations=best_params["match_max_iterations"],
        num_threads=4,
        use_forward_checking=best_params["match_use_forward_checking"],
        use_conflict_backjumping=best_params["match_use_conflict_backjumping"],
    )

    # Create monkey patch for custom seed selection
    original_find_mcs = GraphMatcher.find_maximum_common_subgraph

    def patched_find_mcs(self, seeds=None):
        """Patched version with custom seed parameters"""
        if seeds is None:
            # Create seed selector with our custom parameters
            seed_selector = SeedSelector(
                self.graph_a, self.graph_b, self.compatibility_matrix, seed_params
            )
            seeds = seed_selector.select_seeds()
        return original_find_mcs(self, seeds)

    # Apply monkey patch
    GraphMatcher.find_maximum_common_subgraph = patched_find_mcs

    try:
        # Initialize timer
        start_time = time.time()

        # Create benchmark runner
        runner = BenchmarkRunner(
            datasets=dataset,
            signature_params=signature_params,
            compatibility_params=compatibility_params.__dict__,
            match_params=match_params.__dict__,
            num_workers=4,
            use_processes=True,
        )

        # Run benchmark
        result = runner.run_benchmark(name="Optimized PRISM Benchmark")
        end_time = time.time()

        # Save the results
        results_dir = Path(__file__).parent.parent / "benchmark_results"
        results_path = (
            results_dir / f"optimized_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        result.save_results(str(results_path))

        # Print summary
        print(
            f"\nValidation benchmark completed in {end_time - start_time:.2f} seconds"
        )
        print(f"Results saved to {results_path}")

        # Display summary metrics
        if result.metrics:
            print("\n=== Validation Summary ===")
            print(f"Number of tests: {len(result.test_results)}")
            print(f"Average Match Size: {result.metrics.avg_match_size:.4f}")
            print(f"Average Match Ratio: {result.metrics.avg_match_ratio:.4f}")
            print(f"Average Time: {result.metrics.avg_time:.6f} seconds")

            # Print category breakdown
            if hasattr(result.metrics, "by_category"):
                print("\n=== Performance by Category ===")
                for category, metrics in result.metrics.by_category.items():
                    print(f"{category}:")
                    print(f"  Avg Match Size: {metrics.get('avg_match_size', 0.0):.4f}")
                    print(
                        f"  Avg Match Ratio: {metrics.get('avg_match_ratio', 0.0):.4f}"
                    )
                    print(f"  Avg Time: {metrics.get('avg_time', 0.0):.6f} seconds")

            # Update config file with validated metrics
            save_config_file(
                best_params, result.metrics.avg_match_size, result.metrics.avg_time
            )

        # Restore original method
        GraphMatcher.find_maximum_common_subgraph = original_find_mcs

        # Return validation metrics
        validation_results = {}
        if result.metrics:
            validation_results = {
                "avg_match_size": result.metrics.avg_match_size,
                "avg_match_ratio": result.metrics.avg_match_ratio,
                "avg_time": result.metrics.avg_time,
                "total_time": end_time - start_time,
            }

        return validation_results

    except Exception as e:
        print(f"Error in validation: {e}")
        # Restore original method in case of error
        GraphMatcher.find_maximum_common_subgraph = original_find_mcs
        return {"error": str(e)}


def main():
    """Run hyperparameter optimization for the PRISM algorithm."""
    print("=== PRISM Hyperparameter Optimization ===")

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for PRISM"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of optimization trials"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after optimization"
    )
    args = parser.parse_args()

    # Check for Optuna dependency
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required for hyperparameter optimization.")
        print("Install with: pip install optuna plotly kaleido")
        sys.exit(1)

    # Run optimization
    best_params = run_optimization(args.dataset, n_trials=args.trials)

    # Validate best parameters
    if args.validate:
        validation_results = validate_best_params(args.dataset, best_params)

        # Save validation results
        results_dir = Path(__file__).parent.parent / "optimization_results"
        validation_file = (
            results_dir / f"validation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(validation_file, "w") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "validation_results": validation_results,
                },
                f,
                indent=2,
            )

        print(f"\nValidation results saved to {validation_file}")


if __name__ == "__main__":
    main()
