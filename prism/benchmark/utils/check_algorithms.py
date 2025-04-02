# examples/check_algorithms.py
"""Utility script to check available algorithm adapters and diagnose any issues."""

import os
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from typing import Dict, List, Any

from prism.benchmark.adapters.algorithm_adapters import get_all_algorithm_adapters
from prism.benchmark.core.metrics import BenchmarkMetrics


def check_algorithm_adapters(verbose: bool = False) -> None:
    """Check all algorithm adapters and diagnose any issues.

    Args:
        verbose: Whether to print detailed information about each adapter
    """
    print("Checking algorithm adapters...")

    # Define expected algorithms
    expected_algorithms = ["PRISM", "RDKit-MCS", "VF2", "McGregor"]

    # Default parameters for testing adapters
    test_params = {
        "prism_params": {
            "signature_params": {"max_distance": 3, "use_bond_types": True},
            "compatibility_params": {"element_match_required": True},
            "match_params": {"max_time_seconds": 30},
        },
        "rdkit_params": {"timeout": 30},
        "mcgregor_params": {"timeout": 30},
        "vf2_params": {"timeout": 30},
    }

    # Try to get all adapters
    try:
        algorithms = get_all_algorithm_adapters(**test_params)

        if not algorithms:
            print("Error: No algorithm adapters found")
            print(
                "This could indicate an installation issue with one of the dependencies"
            )
            print("\nTroubleshooting steps:")
            print("1. Ensure all dependencies are installed: rdkit, networkx, etc.")
            print(
                "2. Check if the adapter implementations exist in prism/benchmark/algorithm_adapters.py"
            )
            return

        # Print available algorithms
        print(f"Found {len(algorithms)} adapters:")
        for alg_name, adapter in algorithms.items():
            adapter_type = adapter.__class__.__name__
            availability = "✓ Available"

            if verbose:
                print(f"  - {alg_name} ({adapter_type}): {availability}")
                print(f"    Parameters: {adapter.params}")
            else:
                print(f"  - {alg_name}: {availability}")

        # Check for missing algorithms
        missing_algorithms = [
            alg for alg in expected_algorithms if alg not in algorithms
        ]

        if missing_algorithms:
            print(
                f"\nWarning: The following expected algorithms are missing: {', '.join(missing_algorithms)}"
            )
            print("\nPossible reasons and solutions:")

            for missing_alg in missing_algorithms:
                if missing_alg == "PRISM":
                    print(
                        "- PRISM: Core module may be missing or not properly installed"
                    )
                    print("  Solution: Check your PRISM installation")

                elif missing_alg == "RDKit-MCS":
                    print(
                        "- RDKit-MCS: RDKit may not be installed or properly configured"
                    )
                    print(
                        "  Solution: Run 'pip install rdkit' and ensure it's working correctly"
                    )

                elif missing_alg == "VF2":
                    print("- VF2: NetworkX dependency may be missing")
                    print("  Solution: Run 'pip install networkx'")

                elif missing_alg == "McGregor":
                    print("- McGregor: Custom implementation may be missing")
                    print(
                        "  Solution: Ensure the McGregor adapter is properly implemented"
                    )

            print(
                "\nYou can check the adapter implementations in prism/benchmark/algorithm_adapters.py"
            )
        else:
            print("\nAll expected algorithm adapters are available!")

        # Check metrics compatibility
        print("\nChecking metrics compatibility...")
        try:
            # Create an empty metrics object to verify the module is working
            metrics = BenchmarkMetrics()
            print("✓ Unified metrics system is available")
            print(
                "  You can use the same metrics for both individual and comparative benchmarks"
            )
            print("  For examples of benchmark usage, see:")
            print("  - examples/run_benchmark.py: For single algorithm benchmarking")
            print(
                "  - examples/run_comparative_benchmark.py: For comparing multiple algorithms"
            )
        except Exception as e:
            print(f"Error checking metrics: {e}")
            print("The unified metrics system may not be properly installed.")

    except Exception as e:
        print(f"Error loading algorithm adapters: {e}")
        print(
            "\nThis might indicate a deeper issue with the adapter implementation or dependencies"
        )
        print("Check the adapter code in prism/benchmark/algorithm_adapters.py")


def main():
    """Run the algorithm adapter checker utility."""
    parser = argparse.ArgumentParser(
        description="Check available algorithm adapters and diagnose any issues"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about each adapter",
    )

    args = parser.parse_args()
    check_algorithm_adapters(args.verbose)


if __name__ == "__main__":
    main()
