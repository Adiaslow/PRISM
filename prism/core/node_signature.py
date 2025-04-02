"""Node signature generation for molecular graph preprocessing."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from prism.core.molecular_graph import MolecularGraph


class NodeSignatureGenerator:
    """Generator for advanced node signatures used in isomorphism search."""

    def __init__(
        self,
        max_distance: int = 4,
        use_bond_types: bool = True,
        use_cycles: bool = True,
        element_weight: float = 1.0,
        connectivity_weight: float = 0.8,
        neighborhood_weight: float = 0.6,
        cycles_weight: float = 0.4,
    ):
        """Initialize the node signature generator.

        Args:
            max_distance: Maximum distance to consider for neighborhood signatures.
            use_bond_types: Whether to include bond types in signatures.
            use_cycles: Whether to include cycle information in signatures.
            element_weight: Weight for element type in the signature.
            connectivity_weight: Weight for connectivity in the signature.
            neighborhood_weight: Weight for neighborhood topology in the signature.
            cycles_weight: Weight for cycle participation in the signature.
        """
        self.max_distance = max_distance
        self.use_bond_types = use_bond_types
        self.use_cycles = use_cycles
        self.element_weight = element_weight
        self.connectivity_weight = connectivity_weight
        self.neighborhood_weight = neighborhood_weight
        self.cycles_weight = cycles_weight

        # Element encoding lookup
        self._element_encodings: Dict[str, np.ndarray] = {}

        # Common elements in organic molecules
        self._common_elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

        for i, element in enumerate(self._common_elements):
            # One-hot encoding for common elements
            encoding = np.zeros(len(self._common_elements) + 1)
            encoding[i] = 1.0
            self._element_encodings[element] = encoding

    def generate_signatures(self, graph: MolecularGraph) -> None:
        """Generate signatures for all nodes in the graph.

        Args:
            graph: Molecular graph to generate signatures for.
        """
        for node_id in graph.get_node_ids():
            signature = self._generate_node_signature(graph, node_id)
            graph.set_node_signature(node_id, signature)

    def _generate_node_signature(
        self, graph: MolecularGraph, node_id: int
    ) -> np.ndarray:
        """Generate signature for a specific node.

        Args:
            graph: Molecular graph.
            node_id: ID of the node to generate signature for.

        Returns:
            Signature vector for the node.
        """
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")

        # Element signature component
        element_sig = self._generate_element_signature(node.element)

        # Connectivity signature component
        connectivity_sig = self._generate_connectivity_signature(graph, node_id)

        # Neighborhood topology signature component
        neighborhood_sig = self._generate_neighborhood_signature(graph, node_id)

        # Cycle participation signature component
        if self.use_cycles:
            cycle_sig = self._generate_cycle_signature(graph, node_id)
        else:
            cycle_sig = np.array([])

        # Combine signature components with weights
        signature_parts = [
            self.element_weight * element_sig,
            self.connectivity_weight * connectivity_sig,
            self.neighborhood_weight * neighborhood_sig,
        ]

        if self.use_cycles:
            signature_parts.append(self.cycles_weight * cycle_sig)

        # Concatenate all signature components
        signature = np.concatenate(signature_parts)

        # Normalize the signature vector
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm

        return signature

    def _generate_element_signature(self, element: str) -> np.ndarray:
        """Generate signature component based on element type.

        Args:
            element: Element symbol.

        Returns:
            Element signature vector.
        """
        # Get one-hot encoding for common elements, or use "other" encoding
        if element in self._element_encodings:
            return self._element_encodings[element]
        else:
            # "Other" element encoding
            encoding = np.zeros(len(self._common_elements) + 1)
            encoding[-1] = 1.0
            return encoding

    def _generate_connectivity_signature(
        self, graph: MolecularGraph, node_id: int
    ) -> np.ndarray:
        """Generate signature component based on connectivity.

        Args:
            graph: Molecular graph.
            node_id: ID of the node.

        Returns:
            Connectivity signature vector.
        """
        node = graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")

        # Basic connectivity features
        adjacent_nodes = graph.get_adjacent_nodes(node_id)

        # Count adjacent elements
        element_counts = {element: 0 for element in self._common_elements}
        element_counts["other"] = 0

        for adj_id in adjacent_nodes:
            adj_node = graph.get_node(adj_id)
            if adj_node is None:
                continue

            if adj_node.element in element_counts:
                element_counts[adj_node.element] += 1
            else:
                element_counts["other"] += 1

        # Create connectivity signature
        connectivity_sig = np.array(
            [
                node.degree,  # Degree of the node
                *[
                    element_counts[element] for element in self._common_elements
                ],  # Adjacent element counts
                element_counts["other"],  # Other elements count
            ]
        )

        # Add bond type information if enabled
        if self.use_bond_types:
            # Bond type counts (single, double, triple, aromatic)
            bond_type_counts = {
                "SINGLE": 0,
                "DOUBLE": 0,
                "TRIPLE": 0,
                "AROMATIC": 0,
                "OTHER": 0,
            }

            for adj_id in adjacent_nodes:
                edge = graph.get_edge_between(node_id, adj_id)
                if edge is None:
                    continue

                if edge.bond_type in bond_type_counts:
                    bond_type_counts[edge.bond_type] += 1
                else:
                    bond_type_counts["OTHER"] += 1

            # Append bond type counts to connectivity signature
            bond_sig = np.array(
                [
                    bond_type_counts["SINGLE"],
                    bond_type_counts["DOUBLE"],
                    bond_type_counts["TRIPLE"],
                    bond_type_counts["AROMATIC"],
                    bond_type_counts["OTHER"],
                ]
            )

            connectivity_sig = np.concatenate([connectivity_sig, bond_sig])

        return connectivity_sig

    def _generate_neighborhood_signature(
        self, graph: MolecularGraph, node_id: int
    ) -> np.ndarray:
        """Generate signature component based on neighborhood topology.

        Args:
            graph: Molecular graph.
            node_id: ID of the node.

        Returns:
            Neighborhood signature vector.
        """
        # Count nodes at each distance
        distance_counts = [0] * (self.max_distance + 1)
        visited = {node_id}
        current_level = {node_id}

        for distance in range(1, self.max_distance + 1):
            next_level = set()

            for current_id in current_level:
                for adj_id in graph.get_adjacent_nodes(current_id):
                    if adj_id not in visited:
                        next_level.add(adj_id)
                        visited.add(adj_id)

            distance_counts[distance] = len(next_level)
            current_level = next_level

        # Element distribution at each distance
        element_distance_dist = np.zeros(
            (self.max_distance, len(self._common_elements) + 1)
        )

        # Use BFS to find nodes at each distance
        visited = {node_id}
        current_level = {node_id}

        for distance in range(1, self.max_distance + 1):
            next_level = set()

            for current_id in current_level:
                for adj_id in graph.get_adjacent_nodes(current_id):
                    if adj_id not in visited:
                        next_level.add(adj_id)
                        visited.add(adj_id)

                        # Update element distribution
                        adj_node = graph.get_node(adj_id)
                        if adj_node is None:
                            continue

                        element_idx = self._get_element_index(adj_node.element)
                        element_distance_dist[distance - 1, element_idx] += 1

            current_level = next_level

        # Flatten and normalize the element distribution
        flat_element_dist = element_distance_dist.flatten()

        # Combine with distance counts
        neighborhood_sig = np.concatenate([distance_counts, flat_element_dist])

        return neighborhood_sig

    def _generate_cycle_signature(
        self, graph: MolecularGraph, node_id: int
    ) -> np.ndarray:
        """Generate signature component based on cycle participation.

        Args:
            graph: Molecular graph.
            node_id: ID of the node.

        Returns:
            Cycle signature vector.
        """
        # Get cycles containing the node
        cycles = graph.get_node_cycles(node_id)

        # Count cycles of different sizes
        max_cycle_size = 10  # Consider cycles up to size 10
        cycle_size_counts = [0] * max_cycle_size

        for cycle in cycles:
            size = len(cycle)
            if size <= max_cycle_size:
                cycle_size_counts[size - 1] += 1

        # Create cycle signature
        cycle_sig = np.array(
            [
                len(cycles),  # Total number of cycles
                *cycle_size_counts,  # Counts of cycles by size
            ]
        )

        return cycle_sig

    def _get_element_index(self, element: str) -> int:
        """Get the index of an element in the common elements list.

        Args:
            element: Element symbol.

        Returns:
            Index of the element, or the index of "other" if not found.
        """
        if element in self._common_elements:
            return self._common_elements.index(element)
        else:
            return len(self._common_elements)
