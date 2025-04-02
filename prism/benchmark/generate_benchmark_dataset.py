# examples/generate_benchmark_dataset.py
"""Generate comprehensive benchmark datasets for molecular subgraph isomorphism algorithms.

This script creates a benchmark dataset with controlled variations across key dimensions:
- Molecule sizes (nodes/edges)
- Size of the common substructure
- Number and quality of alternative solutions
- Structural complexity and topology
- Distribution across test categories
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from typing import Dict, List, Optional, Any

from prism.benchmark.services.dataset_generator import DatasetGenerator
from prism.benchmark.configs.benchmark_config import BenchmarkConfig


def generate_benchmark_dataset(
    output_path: str,
    pairs_per_category: int = 5,
    dataset_path: str = "benchmark_data/qm9.csv",
) -> None:
    """Generate a comprehensive benchmark dataset for molecular subgraph isomorphism.

    Args:
        output_path: Path to save the dataset
        pairs_per_category: Number of pairs to generate per category
        dataset_path: Path to the input molecular dataset
    """
    print("Generating molecular benchmark dataset...")

    # Create a benchmark configuration
    config = BenchmarkConfig(
        name="Molecular Benchmark Dataset",
        description="Dataset for evaluating molecular subgraph isomorphism algorithms",
        categories=BenchmarkConfig.default_config().categories,
        pairs_per_category=pairs_per_category,
    )

    # Generate dataset
    generator = DatasetGenerator(config)

    # Load molecular dataset
    print(f"Loading molecular dataset from {dataset_path}...")
    num_loaded = generator.load_molecular_dataset(dataset_path)
    print(f"Loaded {num_loaded} molecules")

    dataset = generator.generate_dataset()

    # Save dataset to temporary file first
    temp_path = output_path + ".temp"
    generator.save_dataset(temp_path)

    # Open and modify the JSON to remove optimal solutions
    try:
        with open(temp_path, "r") as f:
            dataset_json = json.load(f)

        # Remove optimal solutions from all pairs
        for pair in dataset_json.get("pairs", []):
            if "optimal_solution" in pair:
                # Remove the optimal solution
                del pair["optimal_solution"]

        # Save the modified dataset
        with open(output_path, "w") as f:
            json.dump(dataset_json, f, indent=2)

        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Analyze the dataset
        categories = set(
            pair.get("category", "") for pair in dataset_json.get("pairs", [])
        )

        print(f"Dataset saved to {output_path}")
        print(
            f"Generated {len(dataset_json.get('pairs', []))} test cases across {len(categories)} categories:"
        )

        # Show category breakdown
        category_counts = {}
        for pair in dataset_json.get("pairs", []):
            category = pair.get("category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\nCategory Distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} pairs")

        # Show size distribution
        size_ranges = {
            "small": (0, 50),
            "medium": (51, 100),
            "large": (101, 200),
            "xlarge": (201, float("inf")),
        }

        size_counts = {range_name: 0 for range_name in size_ranges}
        for pair in dataset_json.get("pairs", []):
            mol1_size = pair.get("mol1_size", 0)
            mol2_size = pair.get("mol2_size", 0)
            max_size = max(mol1_size, mol2_size)

            for range_name, (min_size, max_range_size) in size_ranges.items():
                if min_size <= max_size <= max_range_size:
                    size_counts[range_name] += 1
                    break

        print("\nMolecule Size Distribution:")
        for range_name, count in size_counts.items():
            print(f"  {range_name}: {count} pairs")

    except Exception as e:
        print(f"Error processing dataset: {e}")
        if os.path.exists(temp_path):
            print(f"Temporary file preserved at {temp_path}")


def main():
    """Generate benchmark dataset for molecular subgraph isomorphism."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark dataset for molecular subgraph isomorphism"
    )
    parser.add_argument(
        "--output",
        default=f"benchmark_results/benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json",
        help="Output path for the dataset",
    )
    parser.add_argument(
        "--pairs", type=int, default=5, help="Number of pairs per category"
    )
    parser.add_argument(
        "--dataset",
        default="benchmark_data/qm9.csv",
        help="Path to the input molecular dataset",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_benchmark_dataset(args.output, args.pairs, args.dataset)


if __name__ == "__main__":
    main()
