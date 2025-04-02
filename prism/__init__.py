"""PRISM: Parallel Recursive Isomorphism Search for Molecules."""

from prism.core.molecular_graph import MolecularGraph
from prism.algorithm.graph_matcher import GraphMatcher, MatchResult, MatchParameters
from prism.core.compatibility_matrix import (
    CompatibilityMatrix,
    CompatibilityParameters,
)
from prism.core.node_signature import NodeSignatureGenerator


class MolecularGraphMatcher:
    """Main interface for molecular graph matching using PRISM algorithm."""

    def __init__(
        self, signature_params=None, compatibility_params=None, match_params=None
    ):
        """Initialize the molecular graph matcher.

        Args:
            signature_params: Optional parameters for node signature generation.
            compatibility_params: Optional parameters for compatibility matrix.
            match_params: Optional parameters for graph matching.
        """
        self.signature_generator = NodeSignatureGenerator(**(signature_params or {}))
        self.compatibility_params = compatibility_params
        self.match_params = match_params

    def find_maximum_common_subgraph(self, mol1, mol2):
        """Find the maximum common subgraph between two molecules.

        Args:
            mol1: First molecule (RDKit Mol object).
            mol2: Second molecule (RDKit Mol object).

        Returns:
            MatchResult: Result containing the mapping and statistics.
        """
        # Convert RDKit molecules to MolecularGraph objects
        graph1 = MolecularGraph.from_rdkit_mol(mol1)
        graph2 = MolecularGraph.from_rdkit_mol(mol2)

        # Generate node signatures
        self.signature_generator.generate_signatures(graph1)
        self.signature_generator.generate_signatures(graph2)

        # Create compatibility matrix
        compatibility_matrix = CompatibilityMatrix(
            graph1, graph2, self.compatibility_params
        )
        compatibility_matrix.refine_matrix()

        # Create graph matcher
        matcher = GraphMatcher(graph1, graph2, compatibility_matrix, self.match_params)

        # Find maximum common subgraph
        result = matcher.find_maximum_common_subgraph()

        return result


__all__ = [
    "MolecularGraphMatcher",
    "MolecularGraph",
    "GraphMatcher",
    "CompatibilityMatrix",
    "NodeSignatureGenerator",
    "MatchResult",
    "MatchParameters",
    "CompatibilityParameters",
]
