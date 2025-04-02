"""Configuration classes for benchmark dataset generation."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CategoryConfig:
    """Configuration for a benchmark category."""

    name: str
    description: str
    size_range: Optional[Dict[str, int]] = None
    topology: Optional[str] = None
    symmetry_level: Optional[str] = None
    challenge_type: Optional[str] = None
    params: Dict[str, any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark dataset generation."""

    name: str
    description: str
    pairs_per_category: int
    categories: Dict[str, CategoryConfig]

    @classmethod
    def default_config(cls) -> "BenchmarkConfig":
        """Create a default benchmark configuration.

        Returns:
            Default benchmark configuration
        """
        return cls(
            name="Default Benchmark",
            description="Default benchmark dataset for testing molecular graph matching algorithms",
            pairs_per_category=10,
            categories={
                "SB-Small": CategoryConfig(
                    name="Size-Balanced Small",
                    description="Small molecules (10-20 atoms) with balanced sizes",
                    size_range={"min": 10, "max": 20},
                ),
                "SB-Medium": CategoryConfig(
                    name="Size-Balanced Medium",
                    description="Medium molecules (20-50 atoms) with balanced sizes",
                    size_range={"min": 20, "max": 50},
                ),
                "SB-Large": CategoryConfig(
                    name="Size-Balanced Large",
                    description="Large molecules (50-100 atoms) with balanced sizes",
                    size_range={"min": 50, "max": 100},
                ),
            },
        )

    @classmethod
    def from_json(cls, file_path: str) -> "BenchmarkConfig":
        """Load benchmark configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Loaded benchmark configuration
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        categories = {k: CategoryConfig(**v) for k, v in data["categories"].items()}

        return cls(
            name=data["name"],
            description=data["description"],
            pairs_per_category=data["pairs_per_category"],
            categories=categories,
        )
