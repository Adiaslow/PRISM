"""Configuration for benchmark datasets."""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field


class SizeRange(BaseModel):
    """Range of sizes for molecular structures."""

    min_nodes: int  # Minimum number of nodes
    max_nodes: int  # Maximum number of nodes

    class Config:
        """Pydantic model configuration."""

        frozen = True


class OverlapConfig(BaseModel):
    """Configuration for overlap between molecules."""

    min_overlap_ratio: float = 0.2  # Minimum ratio of overlap (0.0-1.0)
    max_overlap_ratio: float = 0.9  # Maximum ratio of overlap (0.0-1.0)

    class Config:
        """Pydantic model configuration."""

        frozen = True


class CategoryConfig(BaseModel):
    """Configuration for a benchmark category."""

    # Category identification
    id: str  # Unique identifier for the category
    name: str  # Human-readable name
    description: str  # Description of the category

    # Size parameters
    mol1_size: SizeRange  # Size range for first molecule
    mol2_size: SizeRange  # Size range for second molecule
    overlap: OverlapConfig = Field(
        default_factory=OverlapConfig
    )  # Overlap configuration

    # Complexity parameters
    min_false_leads: int = 1  # Minimum number of significant false leads
    max_false_leads: int = 5  # Maximum number of significant false leads

    # Structural characteristics
    structural_bias: Dict[str, float] = Field(
        default_factory=dict
    )  # Biases toward specific structures

    class Config:
        """Pydantic model configuration."""

        frozen = True


# Pre-defined benchmark categories
SIZE_BALANCED_CATEGORIES = {
    "SB-Small": CategoryConfig(
        id="SB-Small",
        name="Size-Balanced Small",
        description="Small molecules (e.g. amino acids, simple sugars)",
        mol1_size=SizeRange(min_nodes=3, max_nodes=20),
        mol2_size=SizeRange(min_nodes=3, max_nodes=20),
        overlap=OverlapConfig(min_overlap_ratio=0.2, max_overlap_ratio=0.9),
        min_false_leads=1,
        max_false_leads=5,
    ),
    "SB-Medium": CategoryConfig(
        id="SB-Medium",
        name="Size-Balanced Medium",
        description="Medium-sized molecules (e.g. peptides, oligosaccharides)",
        mol1_size=SizeRange(min_nodes=21, max_nodes=50),
        mol2_size=SizeRange(min_nodes=21, max_nodes=50),
        overlap=OverlapConfig(min_overlap_ratio=0.2, max_overlap_ratio=0.9),
        min_false_leads=3,
        max_false_leads=8,
    ),
    "SB-Large": CategoryConfig(
        id="SB-Large",
        name="Size-Balanced Large",
        description="Large molecules (e.g. proteins, complex natural products)",
        mol1_size=SizeRange(min_nodes=51, max_nodes=100),
        mol2_size=SizeRange(min_nodes=51, max_nodes=100),
        overlap=OverlapConfig(min_overlap_ratio=0.2, max_overlap_ratio=0.9),
        min_false_leads=5,
        max_false_leads=12,
    ),
    "SB-XLarge": CategoryConfig(
        id="SB-XLarge",
        name="Size-Balanced Extra Large",
        description="Very large molecules (e.g. macromolecules, polymers)",
        mol1_size=SizeRange(min_nodes=101, max_nodes=200),
        mol2_size=SizeRange(min_nodes=101, max_nodes=200),
        overlap=OverlapConfig(min_overlap_ratio=0.2, max_overlap_ratio=0.9),
        min_false_leads=8,
        max_false_leads=20,
    ),
}

SIZE_IMBALANCED_CATEGORIES = {
    "SI-Small/Medium": CategoryConfig(
        id="SI-Small/Medium",
        name="Size-Imbalanced Small/Medium",
        description="Small molecule in medium-sized molecule",
        mol1_size=SizeRange(min_nodes=5, max_nodes=15),
        mol2_size=SizeRange(min_nodes=20, max_nodes=40),
        overlap=OverlapConfig(min_overlap_ratio=0.4, max_overlap_ratio=0.9),
        min_false_leads=2,
        max_false_leads=6,
    ),
    "SI-Medium/Large": CategoryConfig(
        id="SI-Medium/Large",
        name="Size-Imbalanced Medium/Large",
        description="Medium-sized molecule in large molecule",
        mol1_size=SizeRange(min_nodes=20, max_nodes=40),
        mol2_size=SizeRange(min_nodes=50, max_nodes=80),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.8),
        min_false_leads=4,
        max_false_leads=10,
    ),
    "SI-Small/Large": CategoryConfig(
        id="SI-Small/Large",
        name="Size-Imbalanced Small/Large",
        description="Small molecule in large molecule",
        mol1_size=SizeRange(min_nodes=5, max_nodes=15),
        mol2_size=SizeRange(min_nodes=50, max_nodes=80),
        overlap=OverlapConfig(min_overlap_ratio=0.4, max_overlap_ratio=0.8),
        min_false_leads=3,
        max_false_leads=8,
    ),
    "SI-Tiny/XLarge": CategoryConfig(
        id="SI-Tiny/XLarge",
        name="Size-Imbalanced Tiny/Extra Large",
        description="Tiny molecule in very large molecule",
        mol1_size=SizeRange(min_nodes=3, max_nodes=8),
        mol2_size=SizeRange(min_nodes=100, max_nodes=150),
        overlap=OverlapConfig(min_overlap_ratio=0.5, max_overlap_ratio=1.0),
        min_false_leads=5,
        max_false_leads=15,
    ),
}

TOPOLOGICAL_VARIATION_CATEGORIES = {
    "TV-Linear": CategoryConfig(
        id="TV-Linear",
        name="Topological Variation - Linear",
        description="Linear/chain molecules with low branching factor",
        mol1_size=SizeRange(min_nodes=20, max_nodes=50),
        mol2_size=SizeRange(min_nodes=20, max_nodes=50),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=2,
        max_false_leads=6,
        structural_bias={"linearity": 0.9, "cyclicity": 0.1, "branching": 0.1},
    ),
    "TV-Cyclic": CategoryConfig(
        id="TV-Cyclic",
        name="Topological Variation - Cyclic",
        description="Ring-dominated structures with multiple cycles",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={"linearity": 0.2, "cyclicity": 0.9, "branching": 0.3},
    ),
    "TV-Branched": CategoryConfig(
        id="TV-Branched",
        name="Topological Variation - Branched",
        description="Highly branched structures with high branching factor",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=4,
        max_false_leads=10,
        structural_bias={"linearity": 0.2, "cyclicity": 0.3, "branching": 0.9},
    ),
    "TV-Mixed": CategoryConfig(
        id="TV-Mixed",
        name="Topological Variation - Mixed",
        description="Heterogeneous topologies with combined features",
        mol1_size=SizeRange(min_nodes=30, max_nodes=70),
        mol2_size=SizeRange(min_nodes=30, max_nodes=70),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=10,
        structural_bias={"linearity": 0.5, "cyclicity": 0.5, "branching": 0.5},
    ),
}

SIGNATURE_COMPLEXITY_CATEGORIES = {
    "SC-Low": CategoryConfig(
        id="SC-Low",
        name="Signature Complexity - Low",
        description="Few node types with uniform connectivity",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={"node_diversity": 0.2, "connectivity_diversity": 0.2},
    ),
    "SC-Medium": CategoryConfig(
        id="SC-Medium",
        name="Signature Complexity - Medium",
        description="Moderate node types with semi-diverse connectivity",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={"node_diversity": 0.5, "connectivity_diversity": 0.5},
    ),
    "SC-High": CategoryConfig(
        id="SC-High",
        name="Signature Complexity - High",
        description="Many node types with highly diverse connectivity",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={"node_diversity": 0.9, "connectivity_diversity": 0.9},
    ),
    "SC-Mixed": CategoryConfig(
        id="SC-Mixed",
        name="Signature Complexity - Mixed",
        description="Varied distribution of node types and connectivity",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={
            "node_diversity_variance": 0.8,
            "connectivity_diversity_variance": 0.8,
        },
    ),
}

CHALLENGE_CATEGORIES = {
    "CS-Symmetry": CategoryConfig(
        id="CS-Symmetry",
        name="Challenge Set - Symmetry",
        description="Highly symmetric structures with multiple equivalent solutions",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=5,
        max_false_leads=15,
        structural_bias={"symmetry": 0.9, "equivalent_solutions": 0.9},
    ),
    "CS-NearMiss": CategoryConfig(
        id="CS-NearMiss",
        name="Challenge Set - Near Miss",
        description="Many almost-optimal solutions differing by 1-2 nodes",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=8,
        max_false_leads=15,
        structural_bias={"near_optimal_solutions": 0.9, "solution_size_variance": 0.1},
    ),
    "CS-Bottleneck": CategoryConfig(
        id="CS-Bottleneck",
        name="Challenge Set - Bottleneck",
        description="Solutions requiring passage through constrained intermediate states",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=3,
        max_false_leads=8,
        structural_bias={"bottleneck_topology": 0.9, "path_constraints": 0.9},
    ),
    "CS-Deceptive": CategoryConfig(
        id="CS-Deceptive",
        name="Challenge Set - Deceptive",
        description="Structures with tempting but sub-optimal paths",
        mol1_size=SizeRange(min_nodes=20, max_nodes=60),
        mol2_size=SizeRange(min_nodes=20, max_nodes=60),
        overlap=OverlapConfig(min_overlap_ratio=0.3, max_overlap_ratio=0.7),
        min_false_leads=5,
        max_false_leads=12,
        structural_bias={"deceptive_paths": 0.9, "local_optima": 0.9},
    ),
}

# Combined dictionary of all categories
ALL_CATEGORIES = {
    **SIZE_BALANCED_CATEGORIES,
    **SIZE_IMBALANCED_CATEGORIES,
    **TOPOLOGICAL_VARIATION_CATEGORIES,
    **SIGNATURE_COMPLEXITY_CATEGORIES,
    **CHALLENGE_CATEGORIES,
}


class BenchmarkConfig(BaseModel):
    """Overall configuration for benchmark datasets."""

    name: str  # Name of the benchmark configuration
    description: str  # Description of the benchmark
    categories: Dict[str, CategoryConfig] = Field(
        default_factory=dict
    )  # Categories to include
    pairs_per_category: int = 20  # Number of pairs to generate per category
    seed: Optional[int] = None  # Random seed for reproducibility

    class Config:
        """Pydantic model configuration."""

        frozen = True

    @classmethod
    def default_config(cls) -> "BenchmarkConfig":
        """Create a default benchmark configuration with all categories.

        Returns:
            BenchmarkConfig with all predefined categories
        """
        return cls(
            name="Standard PRISM Benchmark",
            description="Standard benchmark for evaluating PRISM algorithm performance",
            categories=ALL_CATEGORIES,
            pairs_per_category=20,
        )

    @classmethod
    def size_balanced_config(cls) -> "BenchmarkConfig":
        """Create a benchmark configuration with size-balanced categories.

        Returns:
            BenchmarkConfig with size-balanced categories
        """
        return cls(
            name="Size-Balanced PRISM Benchmark",
            description="Benchmark with size-balanced molecular pairs",
            categories=SIZE_BALANCED_CATEGORIES,
            pairs_per_category=20,
        )

    @classmethod
    def size_imbalanced_config(cls) -> "BenchmarkConfig":
        """Create a benchmark configuration with size-imbalanced categories.

        Returns:
            BenchmarkConfig with size-imbalanced categories
        """
        return cls(
            name="Size-Imbalanced PRISM Benchmark",
            description="Benchmark with size-imbalanced molecular pairs",
            categories=SIZE_IMBALANCED_CATEGORIES,
            pairs_per_category=20,
        )

    @classmethod
    def topological_variation_config(cls) -> "BenchmarkConfig":
        """Create a benchmark configuration with topological variation categories.

        Returns:
            BenchmarkConfig with topological variation categories
        """
        return cls(
            name="Topological Variation PRISM Benchmark",
            description="Benchmark with different topological variations",
            categories=TOPOLOGICAL_VARIATION_CATEGORIES,
            pairs_per_category=20,
        )

    @classmethod
    def signature_complexity_config(cls) -> "BenchmarkConfig":
        """Create a benchmark configuration with signature complexity categories.

        Returns:
            BenchmarkConfig with signature complexity categories
        """
        return cls(
            name="Signature Complexity PRISM Benchmark",
            description="Benchmark with varying signature complexity",
            categories=SIGNATURE_COMPLEXITY_CATEGORIES,
            pairs_per_category=20,
        )

    @classmethod
    def challenge_config(cls) -> "BenchmarkConfig":
        """Create a benchmark configuration with challenge categories.

        Returns:
            BenchmarkConfig with challenging test cases
        """
        return cls(
            name="Challenge PRISM Benchmark",
            description="Benchmark with challenging test cases",
            categories=CHALLENGE_CATEGORIES,
            pairs_per_category=20,
        )
