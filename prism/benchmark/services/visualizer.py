"""Visualization utilities for benchmark results."""

import os
import time
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from prism.benchmark.core.implementations.benchmark import BenchmarkResult


class BenchmarkVisualizer:
    """Visualizer for benchmark results."""

    def __init__(self, result: Union[BenchmarkResult, str]):
        """Initialize the visualizer with benchmark results.

        Args:
            result: Either a BenchmarkResult object or a path to a JSON result file
        """
        # Load result if string is provided
        if isinstance(result, str):
            self.result = BenchmarkResult.load_results(result)
        else:
            self.result = result

        # Convert to DataFrame for easier analysis
        self.df = self.result.to_dataframe()

    def save_plots(self, output_dir: str = "benchmark_plots", prefix: str = "") -> None:
        """Generate and save all plots to the output directory.

        Args:
            output_dir: Directory to save plots
            prefix: Prefix for all plot filenames
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save all plots
        self.plot_time_by_size(
            save_path=os.path.join(output_dir, f"{prefix}time_by_size.png")
        )
        self.plot_time_by_category(
            save_path=os.path.join(output_dir, f"{prefix}time_by_category.png")
        )
        self.plot_accuracy_by_category(
            save_path=os.path.join(output_dir, f"{prefix}accuracy_by_category.png")
        )
        self.plot_f1_by_category(
            save_path=os.path.join(output_dir, f"{prefix}f1_by_category.png")
        )
        self.plot_metrics_heatmap(
            save_path=os.path.join(output_dir, f"{prefix}metrics_heatmap.png")
        )
        self.plot_size_comparison(
            save_path=os.path.join(output_dir, f"{prefix}size_comparison.png")
        )

        # Generate summary report
        self.generate_summary_report(
            save_path=os.path.join(output_dir, f"{prefix}summary_report.txt")
        )

    def plot_time_by_size(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot execution time by problem size.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        plt.figure(figsize=(10, 6))

        # Create scatter plot
        sns.scatterplot(
            data=self.df,
            x="problem_size",
            y="time",
            hue="category",
            size="mol1_size",
            sizes=(20, 200),
            alpha=0.7,
        )

        # Add trend line
        sns.regplot(
            data=self.df,
            x="problem_size",
            y="time",
            scatter=False,
            line_kws={"color": "red", "linestyle": "--"},
        )

        # Set labels and title
        plt.xlabel("Problem Size (Total Nodes)")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Execution Time by Problem Size")
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_time_by_category(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot execution time by category as a box plot.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        plt.figure(figsize=(12, 6))

        # Create box plot
        ax = sns.boxplot(data=self.df, x="category", y="time", palette="viridis")

        # Add swarm plot for individual points
        sns.swarmplot(
            data=self.df, x="category", y="time", color="black", alpha=0.5, size=4
        )

        # Set labels and title
        plt.xlabel("Category")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Execution Time Distribution by Category")
        plt.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_accuracy_by_category(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot accuracy by category.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        # Group by category and calculate mean metrics
        category_metrics = (
            self.df.groupby("category")
            .agg(
                {
                    "exact_match": "mean",
                    "precision": "mean",
                    "recall": "mean",
                    "f1": "mean",
                }
            )
            .reset_index()
        )

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Create bar plot for exact match accuracy
        sns.barplot(
            data=category_metrics,
            x="category",
            y="exact_match",
            palette="Blues_d",
            alpha=0.7,
            label="Exact Match",
        )

        # Set labels and title
        plt.xlabel("Category")
        plt.ylabel("Accuracy (% Exact Matches)")
        plt.title("Solution Accuracy by Category")
        plt.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add value labels on top of bars
        for i, row in enumerate(category_metrics.itertuples()):
            plt.text(
                i,
                row.exact_match + 0.02,
                f"{row.exact_match:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add reference line for overall accuracy
        overall_accuracy = self.df["exact_match"].mean()
        plt.axhline(
            y=overall_accuracy,
            color="red",
            linestyle="--",
            label=f"Overall Accuracy: {overall_accuracy:.2f}",
        )

        # Add legend
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_f1_by_category(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot F1 score by category.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        # Group by category and calculate mean metrics
        category_metrics = (
            self.df.groupby("category")
            .agg({"precision": "mean", "recall": "mean", "f1": "mean"})
            .reset_index()
        )

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Create bar plots for precision, recall, and F1
        x = np.arange(len(category_metrics))
        width = 0.25

        plt.bar(
            x - width,
            category_metrics["precision"],
            width,
            label="Precision",
            color="skyblue",
        )
        plt.bar(
            x, category_metrics["recall"], width, label="Recall", color="lightgreen"
        )
        plt.bar(
            x + width, category_metrics["f1"], width, label="F1 Score", color="salmon"
        )

        # Set labels and title
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.title("Precision, Recall, and F1 Score by Category")
        plt.xticks(x, category_metrics["category"], rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # Add legend
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_metrics_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot a heatmap of performance metrics by category.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        # Group by category and calculate metrics
        metrics_by_category = self.df.groupby("category").agg(
            {
                "exact_match": "mean",
                "precision": "mean",
                "recall": "mean",
                "f1": "mean",
                "time": "mean",
            }
        )

        # Normalize time column for better visualization
        max_time = metrics_by_category["time"].max()
        metrics_by_category["time_normalized"] = 1 - (
            metrics_by_category["time"] / max_time
        )

        # Select columns for heatmap
        heatmap_data = metrics_by_category[
            ["exact_match", "precision", "recall", "f1", "time_normalized"]
        ]

        # Rename columns for display
        heatmap_data = heatmap_data.rename(
            columns={
                "exact_match": "Exact Match",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1 Score",
                "time_normalized": "Speed",
            }
        )

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Score (higher is better)"},
        )

        # Set title
        plt.title("Performance Metrics by Category")

        # Rotate y-axis labels for better readability
        plt.yticks(rotation=0)

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def plot_size_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of result size vs. optimal size.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure
        """
        plt.figure(figsize=(10, 6))

        # Create scatter plot
        sns.scatterplot(
            data=self.df,
            x="optimal_size",
            y="result_size",
            hue="category",
            style="category",
            s=80,
            alpha=0.7,
        )

        # Add reference line for perfect match
        max_size = max(self.df["optimal_size"].max(), self.df["result_size"].max())
        plt.plot([0, max_size], [0, max_size], "r--", label="Perfect Match")

        # Set labels and title
        plt.xlabel("Optimal Solution Size")
        plt.ylabel("Algorithm Result Size")
        plt.title("Comparison of Result Size vs. Optimal Size")
        plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return plt.gcf()

    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """Generate a summary report of benchmark results.

        Args:
            save_path: Path to save the report (if None, report is only returned)

        Returns:
            Report text as a string
        """
        # Create a list of report lines
        report_lines = [
            "=== PRISM Benchmark Summary Report ===",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {self.result.dataset_name if hasattr(self.result, 'dataset_name') else 'Unknown'}",
            f"Total test cases: {len(self.df)}",
            "",
            "=== Overall Performance ===",
        ]

        # Get overall metrics if available
        if hasattr(self.result, "metrics") and self.result.metrics is not None:
            try:
                overall_metrics = self.result.metrics.dict()

                # Add basic metrics
                report_lines.append(
                    f"Accuracy: {overall_metrics.get('accuracy', 0.0):.4f}"
                )

                # Add avg_score if present
                if "avg_score" in overall_metrics:
                    report_lines.append(
                        f"Average Score: {overall_metrics['avg_score']:.4f}"
                    )

                # Add other performance metrics
                report_lines.extend(
                    [
                        f"Average F1 Score: {overall_metrics.get('avg_f1', 0.0):.4f}",
                        f"Average Precision: {overall_metrics.get('avg_precision', 0.0):.4f}",
                        f"Average Recall: {overall_metrics.get('avg_recall', 0.0):.4f}",
                        "",
                        "=== Timing Performance ===",
                        f"Average Time: {overall_metrics.get('avg_time', 0.0):.4f} seconds",
                        f"Median Time: {overall_metrics.get('median_time', 0.0):.4f} seconds",
                        f"Min Time: {overall_metrics.get('min_time', 0.0):.4f} seconds",
                        f"Max Time: {overall_metrics.get('max_time', 0.0):.4f} seconds",
                        f"Time Std Dev: {overall_metrics.get('time_std', 0.0):.4f} seconds",
                    ]
                )
            except Exception as e:
                report_lines.append(f"Error calculating metrics: {str(e)}")
        else:
            report_lines.append("No metrics available")

        report_lines.append("")

        # Join lines
        report_text = "\n".join(report_lines)

        # Save if path is provided
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)

        return report_text
