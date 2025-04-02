# prism/benchmark/metrics.py
"""Metrics for analyzing and comparing molecular graph matching algorithms."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from prism.algorithm.graph_matcher import MatchResult


class BenchmarkMetrics(BaseModel):
    """Metrics for evaluating and comparing algorithm performance."""

    # Match size metrics
    avg_match_size: float = 0.0  # Average number of nodes in the match
    avg_match_ratio: float = 0.0  # Average ratio of match size to min molecule size
    max_match_size: int = 0  # Maximum match size found
    min_match_size: int = 0  # Minimum match size found

    # Performance metrics
    avg_time: float = 0.0  # Average time in seconds
    median_time: float = 0.0  # Median time in seconds
    min_time: float = 0.0  # Minimum time in seconds
    max_time: float = 0.0  # Maximum time in seconds
    time_std: float = 0.0  # Standard deviation of times

    # Scaling metrics
    size_to_time_correlation: float = 0.0  # Correlation between problem size and time
    size_to_match_correlation: float = (
        0.0  # Correlation between problem size and match size
    )

    # Memory metrics (if available)
    avg_memory: Optional[float] = None  # Average memory usage in MB
    max_memory: Optional[float] = None  # Maximum memory usage in MB

    # Quality metrics (when reference solutions are available)
    accuracy: Optional[float] = None  # Percentage of correct results

    # Match quality estimation (without reference solutions)
    common_substructure_ratio: float = (
        0.0  # Average ratio of match size to max common size estimate
    )

    # Category-specific metrics
    by_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Size-based metrics
    by_size_range: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Additional metrics
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a flat dictionary format.

        Returns:
            Dictionary representation of the metrics
        """
        result = {
            "avg_match_size": self.avg_match_size,
            "avg_match_ratio": self.avg_match_ratio,
            "max_match_size": self.max_match_size,
            "min_match_size": self.min_match_size,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "time_std": self.time_std,
            "size_to_time_correlation": self.size_to_time_correlation,
            "size_to_match_correlation": self.size_to_match_correlation,
            "common_substructure_ratio": self.common_substructure_ratio,
        }

        # Add reference-based metrics if available
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy

        # Add memory metrics if available
        if self.avg_memory is not None:
            result["avg_memory"] = self.avg_memory
        if self.max_memory is not None:
            result["max_memory"] = self.max_memory

        # Add category-specific metrics
        for category, metrics in self.by_category.items():
            for metric_name, metric_value in metrics.items():
                result[f"{category}_{metric_name}"] = metric_value

        # Add size-range metrics
        for size_range, metrics in self.by_size_range.items():
            for metric_name, metric_value in metrics.items():
                result[f"{size_range}_{metric_name}"] = metric_value

        # Add additional metrics
        for metric_name, metric_value in self.additional_metrics.items():
            result[metric_name] = metric_value

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame.

        Returns:
            DataFrame representation of the metrics
        """
        return pd.DataFrame([self.to_dict()])

    def to_json(self, file_path: str) -> None:
        """Save metrics to a JSON file.

        Args:
            file_path: Path to save the metrics
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_results(
        cls, test_results: List[Dict[str, Any]], categories: Optional[List[str]] = None
    ) -> "BenchmarkMetrics":
        """Calculate metrics from a list of test results.

        Args:
            test_results: List of test result dictionaries
            categories: Optional list of categories to include in the metrics

        Returns:
            BenchmarkMetrics calculated from the results
        """
        # Extract basic metrics
        times = [result.get("time", 0.0) for result in test_results]
        match_sizes = [result.get("result_size", 0) for result in test_results]

        # Extract problem sizes and match ratios
        problem_sizes = [result.get("problem_size", 0) for result in test_results]
        min_mol_sizes = [
            min(result.get("mol1_size", 1), result.get("mol2_size", 1))
            for result in test_results
        ]
        match_ratios = [
            match_size / min_mol_size if min_mol_size > 0 else 0.0
            for match_size, min_mol_size in zip(match_sizes, min_mol_sizes)
        ]

        # Handle exact_match with potential None values
        exact_matches = []
        for result in test_results:
            match_value = result.get("exact_match")
            if match_value is not None:
                exact_matches.append(bool(match_value))

        # Memory metrics if available
        memories = [
            float(result.get("memory_mb", 0.0))
            for result in test_results
            if "memory_mb" in result and result.get("memory_mb") is not None
        ]

        # Calculate overall metrics with explicit float conversion
        metrics = cls(
            avg_match_size=float(np.mean(match_sizes) if match_sizes else 0.0),
            avg_match_ratio=float(np.mean(match_ratios) if match_ratios else 0.0),
            max_match_size=int(max(match_sizes) if match_sizes else 0),
            min_match_size=int(min(match_sizes) if match_sizes else 0),
            avg_time=float(np.mean(times) if times else 0.0),
            median_time=float(np.median(times) if times else 0.0),
            min_time=float(min(times) if times else 0.0),
            max_time=float(max(times) if times else 0.0),
            time_std=float(np.std(times) if times else 0.0),
        )

        # Add reference-based metrics if available
        if exact_matches:
            metrics.accuracy = float(
                sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
            )

        # Calculate correlation between problem size and time
        if problem_sizes and len(problem_sizes) > 1 and len(times) > 1:
            metrics.size_to_time_correlation = float(
                np.corrcoef(problem_sizes, times)[0, 1]
            )

        # Calculate correlation between problem size and match size
        if problem_sizes and len(problem_sizes) > 1 and len(match_sizes) > 1:
            metrics.size_to_match_correlation = float(
                np.corrcoef(problem_sizes, match_sizes)[0, 1]
            )

        # Add memory metrics if available
        if memories:
            metrics.avg_memory = (
                float(sum(memories) / len(memories)) if memories else 0.0
            )
            metrics.max_memory = float(max(memories)) if memories else 0.0
        else:
            metrics.avg_memory = 0.0
            metrics.max_memory = 0.0

        # Calculate common substructure ratio using max_common_size when available
        max_common_sizes = [
            result.get(
                "max_common_size",
                min(result.get("mol1_size", 1), result.get("mol2_size", 1)),
            )
            for result in test_results
        ]
        common_ratios = [
            (
                match_size / max_common_size
                if max_common_size and max_common_size > 0
                else 0.0
            )
            for match_size, max_common_size in zip(match_sizes, max_common_sizes)
        ]
        metrics.common_substructure_ratio = float(
            np.mean(common_ratios) if common_ratios else 0.0
        )

        # Calculate category-specific metrics
        if categories:
            for category in categories:
                category_results = [
                    result
                    for result in test_results
                    if result.get("category") == category
                ]

                if category_results:
                    cat_times = [result.get("time", 0.0) for result in category_results]
                    cat_match_sizes = [
                        result.get("result_size", 0) for result in category_results
                    ]
                    cat_min_mol_sizes = [
                        min(result.get("mol1_size", 1), result.get("mol2_size", 1))
                        for result in category_results
                    ]
                    cat_match_ratios = [
                        match_size / min_mol_size if min_mol_size > 0 else 0.0
                        for match_size, min_mol_size in zip(
                            cat_match_sizes, cat_min_mol_sizes
                        )
                    ]

                    # Create category metrics dict
                    category_metrics = {
                        "avg_match_size": float(
                            np.mean(cat_match_sizes) if cat_match_sizes else 0.0
                        ),
                        "avg_match_ratio": float(
                            np.mean(cat_match_ratios) if cat_match_ratios else 0.0
                        ),
                        "avg_time": float(np.mean(cat_times) if cat_times else 0.0),
                    }

                    # Add reference-based metrics if available
                    cat_exact_matches = [
                        result.get("exact_match", False)
                        for result in category_results
                        if result.get("exact_match") is not None
                    ]

                    if cat_exact_matches:
                        category_metrics["accuracy"] = float(
                            sum(cat_exact_matches) / len(cat_exact_matches)
                            if cat_exact_matches
                            else 0.0
                        )

                    metrics.by_category[category] = category_metrics

        # Calculate size-based metrics
        # Group by problem size ranges based on real molecular size categories
        size_ranges = {
            "small": (0, 20),  # Small molecules (amino acids, simple sugars)
            "medium": (21, 50),  # Medium molecules (peptides, oligosaccharides)
            "large": (51, 100),  # Large molecules (proteins, complex natural products)
            "xlarge": (101, 200),  # Very large molecules (macromolecules, polymers)
        }

        for range_name, (min_size, max_size) in size_ranges.items():
            range_results = [
                result
                for result in test_results
                if min_size <= result.get("problem_size", 0) <= max_size
            ]

            if range_results:
                range_times = [result.get("time", 0.0) for result in range_results]
                range_match_sizes = [
                    result.get("result_size", 0) for result in range_results
                ]
                range_min_mol_sizes = [
                    min(result.get("mol1_size", 1), result.get("mol2_size", 1))
                    for result in range_results
                ]
                range_match_ratios = [
                    match_size / min_mol_size if min_mol_size > 0 else 0.0
                    for match_size, min_mol_size in zip(
                        range_match_sizes, range_min_mol_sizes
                    )
                ]

                # Create range metrics dict
                range_metrics = {
                    "avg_match_size": float(
                        np.mean(range_match_sizes) if range_match_sizes else 0.0
                    ),
                    "avg_match_ratio": float(
                        np.mean(range_match_ratios) if range_match_ratios else 0.0
                    ),
                    "avg_time": float(np.mean(range_times) if range_times else 0.0),
                }

                # Add reference-based metrics if available
                range_exact_matches = [
                    result.get("exact_match", False)
                    for result in range_results
                    if result.get("exact_match") is not None
                ]

                if range_exact_matches:
                    range_metrics["accuracy"] = float(
                        sum(range_exact_matches) / len(range_exact_matches)
                        if range_exact_matches
                        else 0.0
                    )

                metrics.by_size_range[range_name] = range_metrics

        return metrics


class TestResult(BaseModel):
    """Result of a single benchmark test for molecular graph matching."""

    # Test identification
    test_id: str  # Unique identifier for the test
    category: str  # Benchmark category

    # Input properties
    mol1_size: int  # Size of the first molecule
    mol2_size: int  # Size of the second molecule
    problem_size: int  # Combined size metric for the problem
    max_common_size: Optional[int] = (
        None  # Size of the maximum common substructure (if known)
    )

    # Performance metrics
    time: float  # Time taken in seconds
    memory_mb: Optional[float] = None  # Memory usage in MB

    # Result metrics
    result_size: int  # Size of the result (number of matched nodes)
    match_ratio: Optional[float] = None  # Ratio of match size to minimum molecule size

    # Quality metrics (when reference solution is available)
    exact_match: Optional[bool] = (
        None  # Whether the result exactly matches a known solution
    )

    # Additional properties
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict
    )  # Algorithm parameters
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict
    )  # Additional metrics

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
