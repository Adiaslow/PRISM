"""Molecular pair representation for benchmarking."""

from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
from pydantic import BaseModel, Field
from rdkit import Chem

from prism.core.molecular_graph import MolecularGraph


class KnownSolution(BaseModel):
    """Known solution for a benchmark test case."""

    mapping: Dict[int, int] = Field(default_factory=dict)  # Mapping from mol1 to mol2
    size: int = 0  # Number of nodes in the solution
    score: float = 0.0  # Quality score of the solution
    is_optimal: bool = False  # Whether this is the optimal solution

    class Config:
        """Pydantic model configuration."""

        frozen = True


class MolecularPair(BaseModel):
    """Pair of molecules for benchmarking with known solutions."""

    # Identification
    id: str  # Unique identifier for the pair
    category: str  # Benchmark category this pair belongs to

    # Molecule representations
    smiles1: str  # SMILES representation of first molecule
    smiles2: str  # SMILES representation of second molecule
    mol1: Optional[Any] = (
        None  # RDKit molecule object (not stored, but can be populated)
    )
    mol2: Optional[Any] = (
        None  # RDKit molecule object (not stored, but can be populated)
    )

    # Properties
    mol1_size: int  # Number of nodes in first molecule
    mol2_size: int  # Number of nodes in second molecule
    max_common_size: int  # Size of the maximum common substructure
    overlap_ratio: float  # Ratio of max common size to min mol size
    false_leads_count: int  # Number of significant suboptimal solutions

    # Known solutions - only optimal solutions
    optimal_solution: KnownSolution = Field(
        default_factory=KnownSolution
    )  # The optimal solution

    # Structural characteristics
    structural_features: Dict[str, Any] = Field(
        default_factory=dict
    )  # Structural properties for categorization

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True  # Allow RDKit mol objects

    def load_molecules(self) -> Tuple[Chem.Mol, Chem.Mol]:
        """Load RDKit molecules from SMILES strings.

        Returns:
            Tuple of (mol1, mol2) as RDKit Mol objects.
        """
        if self.mol1 is None:
            self.mol1 = Chem.MolFromSmiles(self.smiles1)
        if self.mol2 is None:
            self.mol2 = Chem.MolFromSmiles(self.smiles2)

        return self.mol1, self.mol2

    def create_molecular_graphs(self) -> Tuple[MolecularGraph, MolecularGraph]:
        """Create PRISM MolecularGraph objects from the molecules.

        Returns:
            Tuple of (graph1, graph2) as MolecularGraph objects.
        """
        mol1, mol2 = self.load_molecules()
        graph1 = MolecularGraph.from_rdkit_mol(mol1)
        graph2 = MolecularGraph.from_rdkit_mol(mol2)

        return graph1, graph2

    def calculate_solution_metrics(
        self, result_mapping: Dict[int, int]
    ) -> Dict[str, Any]:
        """Calculate metrics for a solution.

        If the optimal solution is available, it will compare against it.
        Otherwise, it will return metrics based on the size of the mapping.

        Args:
            result_mapping: Mapping from molecule 1 to molecule 2 nodes

        Returns:
            Dictionary of metrics including score and size-based metrics
        """
        if self.optimal_solution and self.optimal_solution.mapping:
            # If optimal solution is available, calculate metrics against it
            overlap = set(result_mapping.items()) & set(
                self.optimal_solution.mapping.items()
            )
            match_size = len(overlap)

            # Calculate the score as the fraction of nodes found in the best match over the total nodes in optimal solution
            score = (
                match_size / len(self.optimal_solution.mapping)
                if self.optimal_solution.mapping
                else 0.0
            )

            # Calculate precision and recall for additional metrics
            precision = match_size / len(result_mapping) if result_mapping else 0.0
            recall = (
                match_size / len(self.optimal_solution.mapping)
                if self.optimal_solution.mapping
                else 0.0
            )

            # Calculate F1 score
            f1 = 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)

            # Check if the result is an exact match to the optimal solution
            exact_match = set(result_mapping.items()) == set(
                self.optimal_solution.mapping.items()
            )

            return {
                "score": score,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact_match": exact_match,
            }
        else:
            # If no optimal solution is available, calculate size-based metrics
            # The match size is the size of the mapping
            match_size = len(result_mapping)

            # Calculate score as a ratio of match size to minimum molecule size
            min_mol_size = min(self.mol1_size, self.mol2_size)
            score = match_size / min_mol_size if min_mol_size > 0 else 0.0

            # For consistency with the optimal case, set these metrics
            # but note they have different interpretations without an optimal solution
            return {
                "score": score,
                "precision": 1.0,  # All nodes in the result are considered correct
                "recall": score,  # Recall is the proportion of the smaller molecule matched
                "f1": 2 * score / (1.0 + score) if score > 0 else 0.0,
                "exact_match": None,  # Can't determine exact match without optimal solution
            }


class BenchmarkDataset(BaseModel):
    """Collection of molecular pairs for benchmarking."""

    name: str  # Name of the dataset
    description: str  # Description of the dataset
    pairs: List[MolecularPair] = Field(default_factory=list)  # List of molecular pairs

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def get_pairs_by_category(self, category: str) -> List[MolecularPair]:
        """Get all pairs belonging to a specific category.

        Args:
            category: Category name to filter by

        Returns:
            List of molecular pairs in the specified category
        """
        return [pair for pair in self.pairs if pair.category == category]

    def get_categories(self) -> List[str]:
        """Get all unique categories in the dataset.

        Returns:
            List of category names
        """
        return list(set(pair.category for pair in self.pairs))

    def get_pair_by_id(self, pair_id: str) -> Optional[MolecularPair]:
        """Get a specific pair by its ID.

        Args:
            pair_id: ID of the pair to retrieve

        Returns:
            MolecularPair if found, None otherwise
        """
        for pair in self.pairs:
            if pair.id == pair_id:
                return pair

        return None
