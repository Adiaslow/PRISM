"""Molecular graph representation for PRISM algorithm."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from rdkit import Chem


class Node(BaseModel):
    """Node class representing an atom in a molecular graph."""

    id: int
    element: str
    degree: int
    features: Dict[str, Union[float, int, str]]

    def __hash__(self) -> int:
        """Hash function for the node.

        Returns:
            int: Hash of the node's id.
        """
        return hash(self.id)

    class Config:
        """Pydantic model configuration."""

        frozen = True


class Edge(BaseModel):
    """Edge class representing a bond in a molecular graph."""

    source: int
    target: int
    bond_type: str
    features: Dict[str, Union[float, int, str]]

    class Config:
        """Pydantic model configuration."""

        frozen = True


class MolecularGraph:
    """Graph representation of a molecule optimized for isomorphism search."""

    def __init__(self):
        """Initialize an empty molecular graph."""
        self._graph = nx.Graph()
        self._nodes: Dict[int, Node] = {}
        self._edges: List[Edge] = []
        self._node_signatures: Dict[int, np.ndarray] = {}
        self._adjacent_nodes: Dict[int, Set[int]] = {}

    @classmethod
    def from_rdkit_mol(cls, mol: Chem.Mol) -> MolecularGraph:
        """Create a MolecularGraph from an RDKit molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            MolecularGraph: A new molecular graph instance.
        """
        if mol is None:
            raise ValueError("Input molecule cannot be None")

        graph = cls()

        # Add nodes
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            element = atom.GetSymbol()
            degree = atom.GetDegree()

            features = {
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "num_explicit_hs": atom.GetNumExplicitHs(),
                "is_in_ring": atom.IsInRing(),
            }

            node = Node(id=atom_id, element=element, degree=degree, features=features)
            graph.add_node(node)

        # Add edges
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())

            features = {
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing(),
                "stereo": str(bond.GetStereo()),
            }

            edge = Edge(
                source=begin_atom,
                target=end_atom,
                bond_type=bond_type,
                features=features,
            )
            graph.add_edge(edge)

        return graph

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add.
        """
        self._nodes[node.id] = node
        self._graph.add_node(node.id)
        self._adjacent_nodes[node.id] = set()

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.

        Args:
            edge: Edge to add.
        """
        self._edges.append(edge)
        self._graph.add_edge(edge.source, edge.target, **edge.features)
        self._graph.edges[edge.source, edge.target]["bond_type"] = edge.bond_type

        # Update adjacency lists
        self._adjacent_nodes[edge.source].add(edge.target)
        self._adjacent_nodes[edge.target].add(edge.source)

    def get_node(self, node_id: int) -> Optional[Node]:
        """Get a node by its ID.

        Args:
            node_id: ID of the node to find.

        Returns:
            Node if found, None otherwise.
        """
        return self._nodes.get(node_id)

    def get_adjacent_nodes(self, node_id: int) -> Set[int]:
        """Get IDs of nodes adjacent to the given node.

        Args:
            node_id: ID of the node.

        Returns:
            Set of adjacent node IDs.
        """
        return self._adjacent_nodes.get(node_id, set())

    def get_edge_between(self, source_id: int, target_id: int) -> Optional[Edge]:
        """Get the edge between two nodes if it exists.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            Edge if found, None otherwise.
        """
        for edge in self._edges:
            if (edge.source == source_id and edge.target == target_id) or (
                edge.source == target_id and edge.target == source_id
            ):
                return edge
        return None

    def get_node_ids(self) -> List[int]:
        """Get all node IDs in the graph.

        Returns:
            List of node IDs.
        """
        return list(self._nodes.keys())

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph.

        Returns:
            Number of nodes.
        """
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph.

        Returns:
            Number of edges.
        """
        return len(self._edges)

    def set_node_signature(self, node_id: int, signature: np.ndarray) -> None:
        """Set the signature for a node.

        Args:
            node_id: ID of the node.
            signature: Node signature vector.
        """
        self._node_signatures[node_id] = signature

    def get_node_signature(self, node_id: int) -> Optional[np.ndarray]:
        """Get the signature for a node.

        Args:
            node_id: ID of the node.

        Returns:
            Node signature vector if set, None otherwise.
        """
        return self._node_signatures.get(node_id)

    def get_all_paths(
        self, max_length: int = 5
    ) -> Dict[Tuple[int, int], List[List[int]]]:
        """Get all paths up to a specified length between all pairs of nodes.

        Args:
            max_length: Maximum path length to consider.

        Returns:
            Dictionary mapping node pairs to lists of paths.
        """
        paths = {}

        for source in self.get_node_ids():
            for target in self.get_node_ids():
                if source == target:
                    continue

                paths[(source, target)] = []

                # Use NetworkX all_simple_paths with cutoff
                for path in nx.all_simple_paths(
                    self._graph, source, target, cutoff=max_length
                ):
                    if (
                        len(path) <= max_length + 1
                    ):  # +1 because path includes endpoints
                        paths[(source, target)].append(path)

        return paths

    def get_shortest_path(self, source_id: int, target_id: int) -> List[int]:
        """Get the shortest path between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            List of node IDs in the path, empty if no path exists.
        """
        try:
            return list(
                nx.shortest_path(self._graph, source=source_id, target=target_id)
            )
        except nx.NetworkXNoPath:
            return []

    def get_shortest_path_length(self, source_id: int, target_id: int) -> float:
        """Get the length of the shortest path between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            Length of the shortest path, or infinity if no path exists.
        """
        try:
            return float(
                nx.shortest_path_length(self._graph, source=source_id, target=target_id)
            )
        except nx.NetworkXNoPath:
            return float("inf")

    def get_node_cycles(self, node_id: int) -> List[List[int]]:
        """Get all cycles containing the given node.

        Args:
            node_id: ID of the node.

        Returns:
            List of cycles, where each cycle is a list of node IDs.
        """
        cycles = []

        # Get all cycles in the graph
        cycle_basis = nx.cycle_basis(self._graph)

        # Filter cycles containing the node
        for cycle in cycle_basis:
            if node_id in cycle:
                cycles.append(cycle)

        return cycles

    def get_subgraph(self, node_ids: List[int]) -> MolecularGraph:
        """Create a subgraph containing only the specified nodes.

        Args:
            node_ids: List of node IDs to include.

        Returns:
            New MolecularGraph instance representing the subgraph.
        """
        subgraph = MolecularGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                subgraph.add_node(self._nodes[node_id])

        # Add edges
        for edge in self._edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge)

        return subgraph
