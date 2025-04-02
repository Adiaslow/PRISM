"""Comparative benchmark runner for evaluating multiple algorithms."""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem

from prism.benchmark.adapters.algorithm_adapters import (
    get_all_algorithm_adapters,
    AlgorithmAdapter,
)
from prism.core.molecular_pair import BenchmarkDataset, MolecularPair
from prism.benchmark.core.metrics import BenchmarkMetrics, TestResult


class AlgorithmResult(BaseModel):
    """Results for a single algorithm in a comparative benchmark."""

    # Algorithm identification
    name: str  # Name of the algorithm
    description: str  # Description of the algorithm

    # Results
    test_results: List[TestResult] = Field(
        default_factory=list
    )  # Individual test results
    metrics: Optional[BenchmarkMetrics] = None  # Aggregated metrics

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True


class ComparativeBenchmarkResult(BaseModel):
    """Results of a comparative benchmark run with multiple algorithms."""

    # Benchmark identification
    name: str  # Name of the benchmark
    description: str  # Description of the benchmark
    timestamp: str  # Timestamp when the benchmark was run

    # Hardware information
    system_info: Dict[str, str] = Field(
        default_factory=dict
    )  # Hardware/system information

    # Algorithm results
    algorithm_results: Dict[str, AlgorithmResult] = Field(default_factory=dict)

    # Dataset information
    dataset_info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def calculate_metrics(self) -> None:
        """Calculate metrics for all algorithms."""
        for algorithm_name, result in self.algorithm_results.items():
            if not result.metrics:
                # Extract categories from test results
                categories = list(set(test.category for test in result.test_results))

                # Convert test results to dictionaries
                result_dicts = [test.dict() for test in result.test_results]

                # Calculate metrics
                result.metrics = BenchmarkMetrics.from_results(result_dicts, categories)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for easy comparison.

        Returns:
            DataFrame with algorithm performance comparison
        """
        # Calculate metrics if not already done
        self.calculate_metrics()

        # Create rows for each algorithm
        rows = []
        for algorithm_name, result in self.algorithm_results.items():
            if result.metrics:
                row = {
                    "algorithm": algorithm_name,
                    "description": result.description,
                    "avg_match_size": result.metrics.avg_match_size,
                    "avg_match_ratio": result.metrics.avg_match_ratio,
                    "avg_time": result.metrics.avg_time,
                    "median_time": result.metrics.median_time,
                    "min_time": result.metrics.min_time,
                    "max_time": result.metrics.max_time,
                    "time_std": result.metrics.time_std,
                    "tests_completed": len(result.test_results),
                }

                # Add memory metrics if available
                if result.metrics.avg_memory is not None:
                    row["avg_memory"] = result.metrics.avg_memory
                if result.metrics.max_memory is not None:
                    row["max_memory"] = result.metrics.max_memory

                rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, file_path: str) -> None:
        """Save comparative benchmark results to a JSON file.

        Args:
            file_path: Path to save the results
        """
        # Calculate metrics if not already done
        self.calculate_metrics()

        # Convert to dictionary
        result_dict = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "dataset_info": self.dataset_info,
            "algorithm_results": {},
        }

        # Add algorithm results
        for algorithm_name, alg_result in self.algorithm_results.items():
            result_dict["algorithm_results"][algorithm_name] = {
                "name": alg_result.name,
                "description": alg_result.description,
                "test_results": [test.dict() for test in alg_result.test_results],
                "metrics": alg_result.metrics.dict() if alg_result.metrics else None,
            }

        # Save to file
        with open(file_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_results(cls, file_path: str) -> "ComparativeBenchmarkResult":
        """Load comparative benchmark results from a JSON file.

        Args:
            file_path: Path to the results file

        Returns:
            Loaded ComparativeBenchmarkResult
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Create result object
        result = cls(
            name=data["name"],
            description=data["description"],
            timestamp=data["timestamp"],
            system_info=data["system_info"],
            dataset_info=data.get("dataset_info", {}),
        )

        # Add algorithm results
        for algorithm_name, alg_data in data["algorithm_results"].items():
            # Convert test results to objects
            test_results = [TestResult(**test) for test in alg_data["test_results"]]

            # Create metrics if available
            metrics = None
            if alg_data.get("metrics"):
                metrics = BenchmarkMetrics(**alg_data["metrics"])

            # Create algorithm result
            alg_result = AlgorithmResult(
                name=alg_data["name"],
                description=alg_data["description"],
                test_results=test_results,
                metrics=metrics,
            )

            result.algorithm_results[algorithm_name] = alg_result

        return result


class ComparativeBenchmarkRunner:
    """Runner for comparing multiple molecular subgraph isomorphism algorithms."""

    def __init__(
        self,
        datasets: Union[BenchmarkDataset, List[BenchmarkDataset]],
        algorithms: Optional[Dict[str, AlgorithmAdapter]] = None,
        algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None,
        num_workers: int = 1,
        use_processes: bool = False,
    ):
        """Initialize the comparative benchmark runner.

        Args:
            datasets: Benchmark dataset(s) to run
            algorithms: Optional dictionary of algorithm name to adapter
            algorithm_params: Optional parameters for each algorithm
            num_workers: Number of worker processes/threads to use
            use_processes: Whether to use processes instead of threads
        """
        # Ensure datasets is a list
        self.datasets = datasets if isinstance(datasets, list) else [datasets]

        # Get all available algorithms if not specified
        if algorithms is None:
            self.algorithms = get_all_algorithm_adapters(**(algorithm_params or {}))
        else:
            self.algorithms = algorithms

        # Execution parameters
        self.num_workers = num_workers
        self.use_processes = use_processes

        # Get system information
        self.system_info = self._get_system_info()

    def run_comparative_benchmark(
        self,
        name: str = "Comparative Benchmark",
        description: str = "Comparison of multiple molecular subgraph isomorphism algorithms",
    ) -> ComparativeBenchmarkResult:
        """Run the comparative benchmark on all datasets with all algorithms.

        Args:
            name: Name for the benchmark run
            description: Description of the benchmark

        Returns:
            ComparativeBenchmarkResult containing results for all algorithms
        """
        # Create result object
        result = ComparativeBenchmarkResult(
            name=name,
            description=description,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.system_info,
            dataset_info={
                "num_datasets": len(self.datasets),
                "dataset_names": [dataset.name for dataset in self.datasets],
                "total_pairs": sum(len(dataset.pairs) for dataset in self.datasets),
            },
        )

        # Get all molecular pairs from all datasets
        all_pairs = []
        for dataset in self.datasets:
            all_pairs.extend(dataset.pairs)

        print(
            f"Running comparative benchmark on {len(all_pairs)} molecular pairs using {len(self.algorithms)} algorithms"
        )

        # Create algorithm results
        for algorithm_name, algorithm in self.algorithms.items():
            print(f"Benchmarking algorithm: {algorithm_name}")

            # Initialize algorithm result
            alg_result = AlgorithmResult(
                name=algorithm.name,
                description=algorithm.description,
                test_results=[],
            )

            # Run benchmark with this algorithm
            # Run in parallel if requested
            if self.num_workers > 1:
                # Use process pool if requested, otherwise thread pool
                executor_class = (
                    ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
                )

                with executor_class(max_workers=self.num_workers) as executor:
                    # Submit all tasks
                    futures = []
                    for pair in all_pairs:
                        futures.append(
                            executor.submit(self._run_single_test, algorithm, pair)
                        )

                    # Collect results
                    for i, future in enumerate(futures):
                        if i % 10 == 0:
                            print(f"  Progress: {i}/{len(futures)} tests complete")

                        try:
                            test_result = future.result()
                            if test_result:
                                alg_result.test_results.append(test_result)
                        except Exception as e:
                            print(f"Error in benchmark test: {e}")
                            traceback.print_exc()
            else:
                # Run sequentially
                for i, pair in enumerate(all_pairs):
                    if i % 10 == 0:
                        print(f"  Progress: {i}/{len(all_pairs)} tests complete")

                    try:
                        test_result = self._run_single_test(algorithm, pair)
                        if test_result:
                            alg_result.test_results.append(test_result)
                    except Exception as e:
                        print(f"Error in benchmark test: {e}")
                        traceback.print_exc()

            # Add to results
            result.algorithm_results[algorithm_name] = alg_result

            print(
                f"  Completed {len(alg_result.test_results)} tests for {algorithm_name}"
            )

        # Calculate metrics
        result.calculate_metrics()

        return result

    def _run_single_test(
        self, algorithm: AlgorithmAdapter, pair: MolecularPair
    ) -> Optional[TestResult]:
        """Run a single benchmark test with a specific algorithm.

        Args:
            algorithm: Algorithm adapter to use
            pair: MolecularPair to test

        Returns:
            TestResult or None if test failed
        """
        # Load molecules
        mol1, mol2 = pair.load_molecules()
        if not mol1 or not mol2:
            print(f"Failed to load molecules for pair {pair.id}")
            return None

        # Measure memory usage (if psutil is available)
        memory_mb = None
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_start = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            # psutil not available
            memory_start = 0

        # Run the algorithm
        algorithm_result = algorithm.find_maximum_common_subgraph(mol1, mol2)

        # Extract results
        mapping = algorithm_result.get("mapping", {})
        size = algorithm_result.get("size", 0)
        exec_time = algorithm_result.get("time", 0.0)
        success = algorithm_result.get("success", False)
        error = algorithm_result.get("error")

        # Measure memory usage
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_end = process.memory_info().rss / (1024 * 1024)  # MB
            memory_mb = memory_end - memory_start
        except ImportError:
            # psutil not available
            pass

        # Calculate match ratio (size relative to smallest molecule)
        min_mol_size = min(pair.mol1_size, pair.mol2_size)
        match_ratio = size / min_mol_size if min_mol_size > 0 else 0.0

        # If not successful, return a failure result
        if not success:
            return TestResult(
                test_id=pair.id,
                category=pair.category,
                mol1_size=pair.mol1_size,
                mol2_size=pair.mol2_size,
                problem_size=pair.mol1_size + pair.mol2_size,
                max_common_size=pair.max_common_size,
                time=exec_time,
                memory_mb=memory_mb,
                result_size=0,
                match_ratio=0.0,
                additional_metrics={"error": error},
            )

        # Calculate quality metrics using the known optimal solution
        solution_metrics = pair.calculate_solution_metrics(mapping)

        # Create test result
        return TestResult(
            test_id=pair.id,
            category=pair.category,
            mol1_size=pair.mol1_size,
            mol2_size=pair.mol2_size,
            problem_size=pair.mol1_size + pair.mol2_size,
            max_common_size=pair.max_common_size,
            time=exec_time,
            memory_mb=memory_mb,
            result_size=size,
            match_ratio=match_ratio,
            precision=solution_metrics["precision"],
            recall=solution_metrics["recall"],
            f1=solution_metrics["f1"],
            exact_match=solution_metrics["exact_match"],
        )

    def _get_system_info(self) -> Dict[str, str]:
        """Get information about the system.

        Returns:
            Dictionary of system information
        """
        info = {}

        # Python version
        import platform

        info["python_version"] = platform.python_version()
        info["platform"] = platform.platform()
        info["processor"] = platform.processor()

        # CPU info
        try:
            import psutil

            info["cpu_count"] = str(psutil.cpu_count(logical=False))
            info["cpu_count_logical"] = str(psutil.cpu_count(logical=True))

            memory = psutil.virtual_memory()
            info["total_memory_gb"] = f"{memory.total / (1024**3):.2f}"
        except ImportError:
            # psutil not available
            pass

        # NumPy version
        info["numpy_version"] = np.__version__

        # RDKit version
        try:
            info["rdkit_version"] = Chem.__version__
        except AttributeError:
            info["rdkit_version"] = "Unknown"

        return info
