# prism/benchmark/services/comparative_visualizer.py
"""Visualization utilities for comparative benchmark results."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from prism.benchmark.services.comparative_benchmark import (
    AlgorithmResult,
    ComparativeBenchmarkResult,
)


class ComparativeVisualizer:
    """Visualizer for comparative benchmark results."""

    def __init__(self, result: ComparativeBenchmarkResult):
        """Initialize the visualizer.

        Args:
            result: Comparative benchmark result to visualize
        """
        self.result = result
        self.df = result.to_dataframe()

    def save_plots(self, output_dir: str) -> None:
        """Save all plots to files.

        Args:
            output_dir: Directory to save plots in
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create plots
        self.plot_time_comparison(
            save_path=os.path.join(output_dir, "time_comparison.png")
        )
        self.plot_match_size_comparison(
            save_path=os.path.join(output_dir, "match_size_comparison.png")
        )
        self.plot_match_ratio_comparison(
            save_path=os.path.join(output_dir, "match_ratio_comparison.png")
        )
        self.plot_time_distribution(
            save_path=os.path.join(output_dir, "time_distribution.png")
        )
        self.plot_size_distribution(
            save_path=os.path.join(output_dir, "size_distribution.png")
        )
        self.plot_category_comparison(
            save_path=os.path.join(output_dir, "category_comparison.png")
        )

    def plot_time_comparison(self, save_path: Optional[str] = None) -> Figure:
        """Plot time comparison between algorithms.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        # Sort all algorithms except PRISM, then concatenate with PRISM at the start
        non_prism_df = self.df[self.df["algorithm"] != "PRISM"].sort_values(
            by="avg_time", ascending=True
        )
        prism_df = self.df[self.df["algorithm"] == "PRISM"]
        sorted_df = pd.concat([prism_df, non_prism_df])
        plt.barh(sorted_df["algorithm"], sorted_df["avg_time"])
        plt.xlabel("Average Time (seconds)")
        plt.ylabel("Algorithm")
        plt.title("Algorithm Performance Comparison")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        return plt.gcf()

    def plot_match_size_comparison(self, save_path: Optional[str] = None) -> Figure:
        """Plot match size comparison between algorithms.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        # Sort all algorithms except PRISM, then concatenate with PRISM at the start
        non_prism_df = self.df[self.df["algorithm"] != "PRISM"].sort_values(
            by="avg_match_size", ascending=False
        )
        prism_df = self.df[self.df["algorithm"] == "PRISM"]
        sorted_df = pd.concat([prism_df, non_prism_df])
        plt.barh(sorted_df["algorithm"], sorted_df["avg_match_size"])
        plt.xlabel("Average Match Size")
        plt.ylabel("Algorithm")
        plt.title("Algorithm Match Size Comparison")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        return plt.gcf()

    def plot_match_ratio_comparison(self, save_path: Optional[str] = None) -> Figure:
        """Plot match ratio comparison between algorithms.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        # Sort all algorithms except PRISM, then concatenate with PRISM at the start
        non_prism_df = self.df[self.df["algorithm"] != "PRISM"].sort_values(
            by="avg_match_ratio", ascending=False
        )
        prism_df = self.df[self.df["algorithm"] == "PRISM"]
        sorted_df = pd.concat([prism_df, non_prism_df])
        plt.barh(sorted_df["algorithm"], sorted_df["avg_match_ratio"])
        plt.xlabel("Average Match Ratio")
        plt.ylabel("Algorithm")
        plt.title("Algorithm Match Ratio Comparison")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        return plt.gcf()

    def plot_time_distribution(self, save_path: Optional[str] = None) -> Figure:
        """Plot time distribution for each algorithm.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        data = []
        labels = []

        # Add PRISM first if it exists
        if "PRISM" in self.result.algorithm_results:
            prism_result = self.result.algorithm_results["PRISM"]
            data.append([test.time for test in prism_result.test_results])
            labels.append("PRISM")

        # Add other algorithms
        for algorithm_name, result in self.result.algorithm_results.items():
            if algorithm_name != "PRISM":
                times = [test.time for test in result.test_results]
                data.append(times)
                labels.append(algorithm_name)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=45)
        plt.ylabel("Time (seconds)")
        plt.title("Algorithm Time Distribution")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        return plt.gcf()

    def plot_size_distribution(self, save_path: Optional[str] = None) -> Figure:
        """Plot match size distribution for each algorithm.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 6))
        data = []
        labels = []

        # Add PRISM first if it exists
        if "PRISM" in self.result.algorithm_results:
            prism_result = self.result.algorithm_results["PRISM"]
            data.append([test.result_size for test in prism_result.test_results])
            labels.append("PRISM")

        # Add other algorithms
        for algorithm_name, result in self.result.algorithm_results.items():
            if algorithm_name != "PRISM":
                sizes = [test.result_size for test in result.test_results]
                data.append(sizes)
                labels.append(algorithm_name)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=45)
        plt.ylabel("Match Size")
        plt.title("Algorithm Match Size Distribution")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        return plt.gcf()

    def plot_category_comparison(self, save_path: Optional[str] = None) -> Figure:
        """Plot mean match sizes by category for each algorithm.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        # Collect data by category
        category_data = []
        for alg_name, alg_result in self.result.algorithm_results.items():
            for test in alg_result.test_results:
                category_data.append(
                    {
                        "algorithm": alg_name,
                        "category": test.category,
                        "match_size": test.result_size,
                    }
                )

        # Convert to DataFrame and calculate means
        df = pd.DataFrame(category_data)
        category_means = df.pivot_table(
            values="match_size", index="category", columns="algorithm", aggfunc="mean"
        )

        # Reorder columns to put PRISM first
        if "PRISM" in category_means.columns:
            cols = ["PRISM"] + [col for col in category_means.columns if col != "PRISM"]
            category_means = category_means[cols]

        # Create the plot
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        # Set up bar positions
        n_categories = len(category_means.index)
        n_algorithms = len(category_means.columns)
        bar_width = 0.8 / n_algorithms
        index = np.arange(n_categories)

        # Plot bars for each algorithm
        for i, algorithm in enumerate(category_means.columns):
            pos = index + (i * bar_width)
            ax.bar(
                pos, category_means[algorithm], bar_width, label=algorithm, alpha=0.8
            )

        # Customize the plot
        ax.set_xlabel("Category")
        ax.set_ylabel("Mean Match Size")
        ax.set_title("Mean Match Size by Category and Algorithm")
        ax.set_xticks(index + ((n_algorithms - 1) * bar_width) / 2)
        ax.set_xticklabels(category_means.index, rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()

        return fig
