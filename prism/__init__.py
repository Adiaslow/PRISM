"""PRISM: Parallel Recursive Isomorphism Search for Molecules."""

from typing import Union

import networkx as nx
from rdkit import Chem

from prism.core.molecular_graph import MolecularGraph
from prism.algorithm.graph_matcher import GraphMatcher, MatchResult, MatchParameters
from prism.core.compatibility_matrix import (
    CompatibilityMatrix,
    CompatibilityParameters,
)
from prism.core.node_signature import NodeSignatureGenerator
from prism.utils.input_handler import convert_to_molecular_graph


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

    def find_maximum_common_subgraph(
        self,
        mol1: Union[str, Chem.Mol, nx.Graph, MolecularGraph],
        mol2: Union[str, Chem.Mol, nx.Graph, MolecularGraph],
    ) -> MatchResult:
        """Find the maximum common subgraph between two molecules.

        Args:
            mol1: First molecule in any supported format:
                - SMILES string
                - RDKit Mol object
                - NetworkX Graph
                - MolecularGraph
            mol2: Second molecule in any supported format

        Returns:
            MatchResult: Result containing the mapping and statistics.

        Raises:
            ValueError: If input format is not supported or conversion fails
        """
        # Convert inputs to MolecularGraph objects
        graph1 = convert_to_molecular_graph(mol1)
        graph2 = convert_to_molecular_graph(mol2)

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
