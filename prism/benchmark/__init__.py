"""Benchmarking framework for the PRISM molecular graph matching algorithm."""

from prism.benchmark.adapters.algorithm_adapters import (
    AlgorithmAdapter,
    McGregorAdapter,
    PRISMAdapter,
    RDKitMCSAdapter,
    VF2Adapter,
    get_all_algorithm_adapters,
)
from prism.benchmark.core.metrics import BenchmarkMetrics, TestResult
from prism.benchmark.services.benchmark import BenchmarkResult, BenchmarkRunner
from prism.benchmark.services.comparative_benchmark import (
    AlgorithmResult,
    ComparativeBenchmarkResult,
    ComparativeBenchmarkRunner,
)
from prism.benchmark.services.dataset_generator import DatasetGenerator
from prism.core.molecular_pair import BenchmarkDataset, MolecularPair

__all__ = [
    # Standard benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "DatasetGenerator",
    "MolecularPair",
    "BenchmarkDataset",
    "BenchmarkMetrics",
    "TestResult",
    # Comparative benchmark
    "AlgorithmAdapter",
    "PRISMAdapter",
    "RDKitMCSAdapter",
    "VF2Adapter",
    "McGregorAdapter",
    "get_all_algorithm_adapters",
    "ComparativeBenchmarkRunner",
    "ComparativeBenchmarkResult",
    "AlgorithmResult",
]
