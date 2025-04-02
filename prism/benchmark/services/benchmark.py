"""Runner for executing benchmark tests."""

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

from prism import MolecularGraphMatcher
from prism.algorithm.graph_matcher import MatchParameters
from prism.core.molecular_pair import BenchmarkDataset, MolecularPair
from prism.benchmark.core.metrics import BenchmarkMetrics, TestResult
from prism.core.molecular_graph import MolecularGraph
from prism.core.compatibility_matrix import CompatibilityParameters
from prism.core.node_signature import NodeSignatureGenerator


class BenchmarkResult(BaseModel):
    """Results of a benchmark run."""

    # Benchmark identification
    name: str  # Name of the benchmark
    description: str  # Description of the benchmark
    timestamp: str  # Timestamp when the benchmark was run

    # Hardware information
    system_info: Dict[str, str] = Field(
        default_factory=dict
    )  # Hardware/system information

    # Algorithm configuration
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict
    )  # Algorithm parameters

    # Results
    test_results: List[TestResult] = Field(
        default_factory=list
    )  # Individual test results
    metrics: Optional[BenchmarkMetrics] = None  # Aggregated metrics

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def calculate_metrics(self) -> BenchmarkMetrics:
        """Calculate metrics from the test results.

        Returns:
            BenchmarkMetrics calculated from the test results
        """
        # Extract categories from test results
        categories = list(set(result.category for result in self.test_results))

        # Convert test results to dictionaries
        result_dicts = [result.dict() for result in self.test_results]

        # Calculate metrics
        self.metrics = BenchmarkMetrics.from_results(result_dicts, categories)
        return self.metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert test results to a pandas DataFrame.

        Returns:
            DataFrame representation of the test results
        """
        return pd.DataFrame([result.dict() for result in self.test_results])

    def save_results(self, file_path: str) -> None:
        """Save benchmark results to a JSON file.

        Args:
            file_path: Path to save the results
        """
        # Calculate metrics if not already done
        if self.metrics is None:
            self.calculate_metrics()

        # Convert to dictionary
        result_dict = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "algorithm_params": self.algorithm_params,
            "test_results": [result.dict() for result in self.test_results],
            "metrics": self.metrics.dict() if self.metrics else None,
        }

        # Save to file
        with open(file_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_results(cls, file_path: str) -> "BenchmarkResult":
        """Load benchmark results from a JSON file.

        Args:
            file_path: Path to the results file

        Returns:
            Loaded BenchmarkResult
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert test results to objects
        test_results = [TestResult(**result) for result in data["test_results"]]

        # Create metrics if available
        metrics = None
        if data.get("metrics"):
            metrics = BenchmarkMetrics(**data["metrics"])

        # Create result
        return cls(
            name=data["name"],
            description=data["description"],
            timestamp=data["timestamp"],
            system_info=data["system_info"],
            algorithm_params=data["algorithm_params"],
            test_results=test_results,
            metrics=metrics,
        )


class BenchmarkRunner:
    """Runner for executing PRISM algorithm benchmarks."""

    def __init__(
        self,
        datasets: Union[BenchmarkDataset, List[BenchmarkDataset]],
        signature_params: Optional[Dict[str, Any]] = None,
        compatibility_params: Optional[Dict[str, Any]] = None,
        match_params: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        use_processes: bool = False,
    ):
        """Initialize the benchmark runner.

        Args:
            datasets: Benchmark dataset(s) to run
            signature_params: Optional parameters for node signature generation
            compatibility_params: Optional parameters for compatibility matrix
            match_params: Optional parameters for graph matching
            num_workers: Number of worker processes/threads to use
            use_processes: Whether to use processes instead of threads
        """
        # Ensure datasets is a list
        self.datasets = datasets if isinstance(datasets, list) else [datasets]

        # Algorithm parameters
        self.signature_params = signature_params or {}
        self.compatibility_params = compatibility_params or {}
        self.match_params = match_params or {}

        # Parallel execution settings
        self.num_workers = num_workers
        self.use_processes = use_processes

        # System information
        self.system_info = self._get_system_info()

    def run_benchmark(self, name: str = "PRISM Benchmark") -> BenchmarkResult:
        """Run the benchmark on all datasets.

        Args:
            name: Name for the benchmark run

        Returns:
            BenchmarkResult containing test results and metrics
        """
        # Initialize result
        result = BenchmarkResult(
            name=name,
            description=f"Benchmark of PRISM algorithm on {len(self.datasets)} datasets",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.system_info,
            algorithm_params={
                "signature_params": self.signature_params,
                "compatibility_params": self.compatibility_params,
                "match_params": self.match_params,
            },
        )

        # Initialize the algorithm with the given parameters
        matcher = MolecularGraphMatcher(
            signature_params=self.signature_params,
            compatibility_params=(
                CompatibilityParameters(**self.compatibility_params)
                if self.compatibility_params
                else None
            ),
            match_params=(
                MatchParameters(**self.match_params) if self.match_params else None
            ),
        )

        # Collect all pairs from all datasets
        all_pairs = []
        for dataset in self.datasets:
            all_pairs.extend(dataset.pairs)

        # Run the benchmark in parallel if requested
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
                        executor.submit(self._run_single_test, matcher, pair)
                    )

                # Collect results
                for future in futures:
                    try:
                        test_result = future.result()
                        if test_result:
                            result.test_results.append(test_result)
                    except Exception as e:
                        print(f"Error in benchmark test: {e}")
                        traceback.print_exc()
        else:
            # Run sequentially
            for pair in all_pairs:
                try:
                    test_result = self._run_single_test(matcher, pair)
                    if test_result:
                        result.test_results.append(test_result)
                except Exception as e:
                    print(f"Error in benchmark test: {e}")
                    traceback.print_exc()

        # Calculate metrics
        result.calculate_metrics()

        return result

    def _run_single_test(
        self, matcher: MolecularGraphMatcher, pair: MolecularPair
    ) -> Optional[TestResult]:
        """Run a single benchmark test.

        Args:
            matcher: MolecularGraphMatcher instance
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
            pass

        # Run the algorithm and measure time
        start_time = time.time()
        match_result = matcher.find_maximum_common_subgraph(mol1, mol2)
        end_time = time.time()

        # Measure memory usage
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_end = process.memory_info().rss / (1024 * 1024)  # MB
            memory_mb = memory_end - memory_start
        except ImportError:
            # psutil not available
            pass

        # Calculate quality metrics using the known optimal solution
        solution_metrics = pair.calculate_solution_metrics(match_result.mapping)

        # Create test result
        result = TestResult(
            test_id=pair.id,
            category=pair.category,
            mol1_size=pair.mol1_size,
            mol2_size=pair.mol2_size,
            problem_size=pair.mol1_size + pair.mol2_size,
            time=end_time - start_time,
            memory_mb=memory_mb,
            result_size=match_result.size,
            optimal_size=pair.max_common_size,
            score=solution_metrics["score"],
            precision=solution_metrics["precision"],
            recall=solution_metrics["recall"],
            f1=solution_metrics["f1"],
            exact_match=solution_metrics["exact_match"],
            algorithm_params={
                "signature_params": self.signature_params,
                "compatibility_params": self.compatibility_params,
                "match_params": self.match_params,
            },
        )

        return result

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

        # RDKit version - fix with safer retrieval
        try:
            from rdkit import rdBase

            info["rdkit_version"] = rdBase.rdkitVersion
        except (ImportError, AttributeError):
            info["rdkit_version"] = "Unknown"

        return info

    def _run_test(self, pair: MolecularPair, algorithm: Callable) -> TestResult:
        """Run a single test case.

        Args:
            pair: The molecular pair to test
            algorithm: The algorithm to test

        Returns:
            TestResult object with the results
        """
        # Create molecular graphs from the pair
        graph1, graph2 = pair.create_molecular_graphs()

        # Measure memory usage before
        memory_start = 0.0
        try:
            import psutil

            process = psutil.Process()
            memory_start = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            pass

        # Measure time
        start_time = time.time()

        # Run the algorithm
        result = algorithm(graph1, graph2)

        # Calculate metrics
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Measure memory usage after
        memory_used = 0.0
        try:
            import psutil

            process = psutil.Process()
            memory_end = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_end - memory_start
        except ImportError:
            pass

        # Extract solution data
        mapping = result.mapping if result else {}
        mapping_size = len(mapping) if mapping else 0

        # Calculate solution metrics
        metrics = pair.calculate_solution_metrics(mapping)

        # Convert mapping to canonical form for comparison
        canonical_mapping = {}
        if mapping:
            # Convert node IDs to strings for consistent comparison
            canonical_mapping = {str(k): str(v) for k, v in mapping.items()}

        return TestResult(
            test_id=pair.id,
            category=pair.category,
            mol1_size=pair.mol1_size,
            mol2_size=pair.mol2_size,
            problem_size=pair.mol1_size + pair.mol2_size,
            time=elapsed_time,
            memory_mb=memory_used,
            result_size=mapping_size,
            optimal_size=pair.max_common_size,
            score=metrics["score"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            exact_match=bool(metrics["exact_match"]),  # Ensure boolean type
            additional_metrics={"mapping": canonical_mapping},
        )
