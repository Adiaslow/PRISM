"""Compatibility matrix generation for molecular graphs."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy import sparse

from prism.core.molecular_graph import MolecularGraph


class CompatibilityParameters(BaseModel):
    """Parameters for controlling compatibility matrix generation."""

    element_match_required: bool = True
    min_signature_similarity: float = 0.6
    bond_type_match_required: bool = False
    progressive_refinement: bool = True
    max_compatible_degree_diff: int = 1
    use_sparse_matrix: bool = True

    class Config:
        """Pydantic model configuration."""

        frozen = True


class CompatibilityMatrix:
    """Compatibility matrix between two molecular graphs."""

    def __init__(
        self,
        graph_a: MolecularGraph,
        graph_b: MolecularGraph,
        parameters: Optional[CompatibilityParameters] = None,
    ):
        """Initialize the compatibility matrix generator.

        Args:
            graph_a: First molecular graph.
            graph_b: Second molecular graph.
            parameters: Optional compatibility parameters.
        """
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.parameters = parameters or CompatibilityParameters()

        # Node IDs in each graph
        self.nodes_a = graph_a.get_node_ids()
        self.nodes_b = graph_b.get_node_ids()

        # Create node ID mappings for matrix indexing
        self.node_a_to_index = {node_id: i for i, node_id in enumerate(self.nodes_a)}
        self.node_b_to_index = {node_id: i for i, node_id in enumerate(self.nodes_b)}

        # Initialize matrix
        self.matrix = self._create_matrix()

    def _create_matrix(self) -> sparse.csr_matrix:
        """Create the initial compatibility matrix.

        Returns:
            Sparse compatibility matrix.
        """
        num_nodes_a = len(self.nodes_a)
        num_nodes_b = len(self.nodes_b)

        # Use dictionary of keys sparse matrix during construction
        if self.parameters.use_sparse_matrix:
            matrix = sparse.dok_matrix((num_nodes_a, num_nodes_b), dtype=np.float32)
        else:
            matrix = np.zeros((num_nodes_a, num_nodes_b), dtype=np.float32)

        # Compute compatibility scores for all node pairs
        for i, node_a_id in enumerate(self.nodes_a):
            node_a = self.graph_a.get_node(node_a_id)
            if node_a is None:
                continue

            for j, node_b_id in enumerate(self.nodes_b):
                node_b = self.graph_b.get_node(node_b_id)
                if node_b is None:
                    continue

                # Check element compatibility
                if (
                    self.parameters.element_match_required
                    and node_a.element != node_b.element
                ):
                    continue

                # Check degree compatibility
                if (
                    abs(node_a.degree - node_b.degree)
                    > self.parameters.max_compatible_degree_diff
                ):
                    continue

                # Calculate compatibility score
                score = self._calculate_compatibility_score(node_a_id, node_b_id)

                # Only include pairs with sufficient similarity
                if score >= self.parameters.min_signature_similarity:
                    matrix[i, j] = score

        # Convert to CSR format for efficient operations
        if self.parameters.use_sparse_matrix:
            return matrix.tocsr()
        return matrix

    def _calculate_compatibility_score(self, node_a_id: int, node_b_id: int) -> float:
        """Calculate compatibility score between two nodes.

        Args:
            node_a_id: ID of node in first graph.
            node_b_id: ID of node in second graph.

        Returns:
            Compatibility score between 0.0 and 1.0.
        """
        # Get nodes
        node_a = self.graph_a.get_node(node_a_id)
        node_b = self.graph_b.get_node(node_b_id)

        if node_a is None or node_b is None:
            return 0.0

        # Element type match is a strong signal
        element_score = 1.0 if node_a.element == node_b.element else 0.0

        # Degree similarity
        max_degree = max(node_a.degree, node_b.degree)
        if max_degree == 0:
            degree_score = 1.0
        else:
            degree_diff = abs(node_a.degree - node_b.degree)
            degree_score = 1.0 - (degree_diff / (max_degree + 1))

        # Node signature similarity (if available)
        signature_score = 0.0
        sig_a = self.graph_a.get_node_signature(node_a_id)
        sig_b = self.graph_b.get_node_signature(node_b_id)

        if sig_a is not None and sig_b is not None:
            # Cosine similarity between signatures
            dot_product = np.dot(sig_a, sig_b)
            norm_a = np.linalg.norm(sig_a)
            norm_b = np.linalg.norm(sig_b)

            if norm_a > 0 and norm_b > 0:
                signature_score = dot_product / (norm_a * norm_b)

        # Combine scores with weights
        weights = [0.4, 0.2, 0.4]  # Element, degree, signature
        total_score = (
            weights[0] * element_score
            + weights[1] * degree_score
            + weights[2] * signature_score
        )

        return total_score

    def get_compatibility(self, node_a_id: int, node_b_id: int) -> float:
        """Get compatibility score between two nodes.

        Args:
            node_a_id: ID of node in first graph.
            node_b_id: ID of node in second graph.

        Returns:
            Compatibility score between 0.0 and 1.0.
        """
        i = self.node_a_to_index.get(node_a_id)
        j = self.node_b_to_index.get(node_b_id)

        if i is None or j is None:
            return 0.0

        if isinstance(self.matrix, sparse.spmatrix):
            return self.matrix[i, j]
        else:
            return self.matrix[i, j]

    def get_compatible_nodes(
        self, node_id: int, from_graph_a: bool = True
    ) -> List[Tuple[int, float]]:
        """Get compatible nodes for a node in one graph.

        Args:
            node_id: ID of the node.
            from_graph_a: Whether the node is from graph_a.

        Returns:
            List of (node_id, score) tuples for compatible nodes in the other graph.
        """
        if from_graph_a:
            i = self.node_a_to_index.get(node_id)
            if i is None:
                return []

            # Get non-zero entries in row i
            if isinstance(self.matrix, sparse.spmatrix):
                row = self.matrix[i].toarray().flatten()
            else:
                row = self.matrix[i]

            return [(self.nodes_b[j], row[j]) for j in range(len(row)) if row[j] > 0]
        else:
            j = self.node_b_to_index.get(node_id)
            if j is None:
                return []

            # Get non-zero entries in column j
            if isinstance(self.matrix, sparse.spmatrix):
                col = self.matrix[:, j].toarray().flatten()
            else:
                col = self.matrix[:, j]

            return [(self.nodes_a[i], col[i]) for i in range(len(col)) if col[i] > 0]

    def refine_matrix(self) -> None:
        """Refine the compatibility matrix based on neighborhood consistency.

        This applies a filter to eliminate incompatible pairs based on
        their neighbors' compatibility.
        """
        if not self.parameters.progressive_refinement:
            return

        updated = True
        iterations = 0
        max_iterations = 5

        while updated and iterations < max_iterations:
            updated = False
            iterations += 1

            if isinstance(self.matrix, sparse.spmatrix):
                # For sparse matrix, work with non-zero entries
                non_zeros = list(zip(*self.matrix.nonzero()))

                for i, j in non_zeros:
                    node_a_id = self.nodes_a[i]
                    node_b_id = self.nodes_b[j]

                    # Check neighborhood compatibility
                    if not self._check_neighborhood_compatibility(node_a_id, node_b_id):
                        self.matrix[i, j] = 0
                        updated = True
            else:
                # For dense matrix, iterate over all entries
                for i in range(len(self.nodes_a)):
                    for j in range(len(self.nodes_b)):
                        if self.matrix[i, j] > 0:
                            node_a_id = self.nodes_a[i]
                            node_b_id = self.nodes_b[j]

                            # Check neighborhood compatibility
                            if not self._check_neighborhood_compatibility(
                                node_a_id, node_b_id
                            ):
                                self.matrix[i, j] = 0
                                updated = True

            # Convert back to CSR if sparse
            if updated and isinstance(self.matrix, sparse.spmatrix):
                self.matrix = sparse.csr_matrix(self.matrix)

    def _check_neighborhood_compatibility(self, node_a_id: int, node_b_id: int) -> bool:
        """Check if node neighborhoods are compatible.

        Args:
            node_a_id: ID of node in first graph.
            node_b_id: ID of node in second graph.

        Returns:
            True if neighborhoods are compatible, False otherwise.
        """
        neighbors_a = self.graph_a.get_adjacent_nodes(node_a_id)
        neighbors_b = self.graph_b.get_adjacent_nodes(node_b_id)

        # If one node has neighbors but the other doesn't, they're incompatible
        if (len(neighbors_a) > 0 and len(neighbors_b) == 0) or (
            len(neighbors_a) == 0 and len(neighbors_b) > 0
        ):
            return False

        # Check if for each neighbor of A, there's at least one compatible neighbor of B
        for neighbor_a in neighbors_a:
            edge_ab = self.graph_a.get_edge_between(node_a_id, neighbor_a)
            if edge_ab is None:
                continue

            # Find at least one compatible neighbor of B
            compatible_neighbor_found = False

            for neighbor_b in neighbors_b:
                edge_cd = self.graph_b.get_edge_between(node_b_id, neighbor_b)
                if edge_cd is None:
                    continue

                # Check bond type compatibility if required
                if (
                    self.parameters.bond_type_match_required
                    and edge_ab.bond_type != edge_cd.bond_type
                ):
                    continue

                # Check if the neighbors are compatible
                if self.get_compatibility(neighbor_a, neighbor_b) > 0:
                    compatible_neighbor_found = True
                    break

            if not compatible_neighbor_found:
                return False

        return True

    def to_numpy(self) -> np.ndarray:
        """Convert the compatibility matrix to a dense NumPy array.

        Returns:
            NumPy array containing the compatibility scores.
        """
        if isinstance(self.matrix, sparse.spmatrix):
            return self.matrix.toarray()
        return self.matrix
