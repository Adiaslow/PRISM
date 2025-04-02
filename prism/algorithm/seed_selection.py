# prism/algorithm/seed_selection.py
"""Seed selection strategies for PRISM algorithm."""

import heapq
import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from prism.core.compatibility_matrix import CompatibilityMatrix
from prism.core.molecular_graph import MolecularGraph


class SeedPriority(BaseModel):
    """Priority parameters for seed selection."""

    signature_uniqueness_weight: float = 0.4
    structural_importance_weight: float = 0.3
    expansion_potential_weight: float = 0.2
    connectivity_weight: float = 0.1

    class Config:
        """Pydantic model configuration."""

        frozen = True


class SeedParameters(BaseModel):
    """Parameters for seed selection."""

    max_seeds: int = 10
    diversity_threshold: float = 0.3
    min_compatibility_score: float = 0.7
    use_weighted_sampling: bool = True
    priority: SeedPriority = Field(default_factory=SeedPriority)

    class Config:
        """Pydantic model configuration."""

        frozen = True


class SeedCandidate(BaseModel):
    """Candidate seed pair for isomorphism search."""

    node_a: int
    node_b: int
    score: float
    upper_bound: Optional[int] = None

    class Config:
        """Pydantic model configuration."""

        frozen = True


class SeedSelector:
    """Selects seed pairs for isomorphism search."""

    def __init__(
        self,
        graph_a: MolecularGraph,
        graph_b: MolecularGraph,
        compatibility_matrix: CompatibilityMatrix,
        parameters: Optional[SeedParameters] = None,
    ):
        """Initialize the seed selector.

        Args:
            graph_a: First molecular graph.
            graph_b: Second molecular graph.
            compatibility_matrix: Compatibility matrix between the graphs.
            parameters: Optional seed selection parameters.
        """
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.compatibility_matrix = compatibility_matrix
        self.parameters = parameters or SeedParameters()

        # Calculate node rankings for both graphs
        self.node_ranks_a = self._calculate_node_rankings(graph_a)
        self.node_ranks_b = self._calculate_node_rankings(graph_b)

    def select_seeds(self) -> List[SeedCandidate]:
        """Select seed pairs for isomorphism search.

        Returns:
            List of seed candidates.
        """
        # Generate potential seed pairs
        candidates = self._generate_seed_candidates()

        # Calculate upper bounds and create new instances
        candidates_with_bounds = []
        for candidate in candidates:
            upper_bound = self._estimate_upper_bound(candidate.node_a, candidate.node_b)
            # Create a new candidate with the upper_bound already set
            candidates_with_bounds.append(
                SeedCandidate(
                    node_a=candidate.node_a,
                    node_b=candidate.node_b,
                    score=candidate.score,
                    upper_bound=upper_bound,
                )
            )

        # Filter and prioritize seeds
        selected_seeds = self._select_diverse_seeds(candidates_with_bounds)

        return selected_seeds

    def _calculate_node_rankings(self, graph: MolecularGraph) -> Dict[int, float]:
        """Calculate ranking scores for all nodes in a graph.

        Args:
            graph: Molecular graph.

        Returns:
            Dictionary mapping node IDs to ranking scores.
        """
        node_ids = graph.get_node_ids()
        rankings = {}

        # Calculate signature uniqueness
        signature_uniqueness = self._calculate_signature_uniqueness(graph)

        # Calculate structural importance (centrality)
        centrality = nx.betweenness_centrality(graph._graph)

        # Calculate expansion potential
        expansion_potential = {}
        for node_id in node_ids:
            # Count neighbors and neighbors of neighbors
            neighbors = graph.get_adjacent_nodes(node_id)
            neighbors_of_neighbors = set()
            for neighbor in neighbors:
                neighbors_of_neighbors.update(graph.get_adjacent_nodes(neighbor))

            # Exclude the node itself and its direct neighbors
            neighbors_of_neighbors -= {node_id}
            neighbors_of_neighbors -= neighbors

            # Score based on local neighborhood size
            expansion_potential[node_id] = (
                len(neighbors) * 1.0 + len(neighbors_of_neighbors) * 0.5
            )

        # Normalize expansion potential
        max_expansion = (
            max(expansion_potential.values()) if expansion_potential else 1.0
        )
        if max_expansion > 0:
            for node_id in expansion_potential:
                expansion_potential[node_id] /= max_expansion

        # Calculate connectivity score (prefer moderately connected nodes)
        connectivity_score = {}
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if node is None:
                continue

            # Connectivity score peaks at moderate values
            # Score = 1.0 - |degree - optimal_degree| / max_degree
            optimal_degree = 3  # Typical value for organic molecules
            max_degree = max(
                node.degree
                for node_id in node_ids
                if graph.get_node(node_id) is not None
            )
            if max_degree > 0:
                connectivity_score[node_id] = (
                    1.0 - abs(node.degree - optimal_degree) / max_degree
                )
            else:
                connectivity_score[node_id] = 1.0

        # Combine all factors with weights
        for node_id in node_ids:
            rankings[node_id] = (
                self.parameters.priority.signature_uniqueness_weight
                * signature_uniqueness.get(node_id, 0.0)
                + self.parameters.priority.structural_importance_weight
                * centrality.get(node_id, 0.0)
                + self.parameters.priority.expansion_potential_weight
                * expansion_potential.get(node_id, 0.0)
                + self.parameters.priority.connectivity_weight
                * connectivity_score.get(node_id, 0.0)
            )

        return rankings

    def _calculate_signature_uniqueness(
        self, graph: MolecularGraph
    ) -> Dict[int, float]:
        """Calculate signature uniqueness scores for all nodes.

        Args:
            graph: Molecular graph.

        Returns:
            Dictionary mapping node IDs to uniqueness scores.
        """
        node_ids = graph.get_node_ids()
        uniqueness_scores = {}

        # Count element occurrences
        element_counts = {}
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if node is None:
                continue

            element = node.element
            element_counts[element] = element_counts.get(element, 0) + 1

        # Calculate uniqueness based on element rarity
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if node is None:
                continue

            element = node.element
            uniqueness_scores[node_id] = 1.0 / element_counts[element]

        # If signatures are available, refine uniqueness with signature similarity
        signatures_available = all(
            graph.get_node_signature(node_id) is not None for node_id in node_ids
        )

        if signatures_available:
            signature_similarities = {}

            for i, node_id_i in enumerate(node_ids):
                sig_i = graph.get_node_signature(node_id_i)
                if sig_i is None:
                    continue

                total_similarity = 0.0
                count = 0

                for j, node_id_j in enumerate(node_ids):
                    if i == j:
                        continue

                    sig_j = graph.get_node_signature(node_id_j)
                    if sig_j is None:
                        continue

                    # Calculate cosine similarity
                    dot_product = np.dot(sig_i, sig_j)
                    norm_i = np.linalg.norm(sig_i)
                    norm_j = np.linalg.norm(sig_j)

                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        total_similarity += similarity
                        count += 1

                if count > 0:
                    avg_similarity = total_similarity / count
                    # Higher uniqueness = lower average similarity
                    signature_similarities[node_id_i] = 1.0 - avg_similarity

            # Combine element-based and signature-based uniqueness
            for node_id in node_ids:
                if node_id in signature_similarities:
                    uniqueness_scores[node_id] = (
                        uniqueness_scores[node_id] * 0.3
                        + signature_similarities[node_id] * 0.7
                    )

        # Normalize uniqueness scores
        max_uniqueness = max(uniqueness_scores.values()) if uniqueness_scores else 1.0
        if max_uniqueness > 0:
            for node_id in uniqueness_scores:
                uniqueness_scores[node_id] /= max_uniqueness

        return uniqueness_scores

    def _generate_seed_candidates(self) -> List[SeedCandidate]:
        """Generate candidate seed pairs based on compatibility.

        Returns:
            List of candidate seed pairs.
        """
        candidates = []

        # Consider all compatible node pairs as potential seeds
        for node_a_id in self.graph_a.get_node_ids():
            for node_b_id, score in self.compatibility_matrix.get_compatible_nodes(
                node_a_id, True
            ):
                if score >= self.parameters.min_compatibility_score:
                    # Calculate combined score based on compatibility and node rankings
                    combined_score = (
                        score * 0.5
                        + self.node_ranks_a.get(node_a_id, 0.0) * 0.25
                        + self.node_ranks_b.get(node_b_id, 0.0) * 0.25
                    )

                    candidates.append(
                        SeedCandidate(
                            node_a=node_a_id, node_b=node_b_id, score=combined_score
                        )
                    )

        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def _estimate_upper_bound(self, node_a_id: int, node_b_id: int) -> int:
        """Estimate upper bound on subgraph size for a seed pair.

        Args:
            node_a_id: ID of node in first graph.
            node_b_id: ID of node in second graph.

        Returns:
            Estimated upper bound on maximum subgraph size.
        """
        # Simple BFS-based upper bound estimation
        visited_a = {node_a_id}
        visited_b = {node_b_id}
        queue_a = [node_a_id]
        queue_b = [node_b_id]

        # Map of node_a -> compatible nodes in B
        compatible_nodes = {}

        # Process nodes in BFS order
        while queue_a:
            current_a = queue_a.pop(0)

            # Get compatible nodes for current_a
            compatible_nodes[current_a] = set()
            for compatible_b, _ in self.compatibility_matrix.get_compatible_nodes(
                current_a, True
            ):
                compatible_nodes[current_a].add(compatible_b)

            # Add unvisited neighbors to the queue
            for neighbor in self.graph_a.get_adjacent_nodes(current_a):
                if neighbor not in visited_a:
                    visited_a.add(neighbor)
                    queue_a.append(neighbor)

        # Count how many nodes from B could potentially match with A
        matchable_b = set()
        for a_node, compatible_b_nodes in compatible_nodes.items():
            matchable_b.update(compatible_b_nodes)

        # Upper bound is the minimum of:
        # 1. Number of nodes in A's connected component
        # 2. Number of potentially matchable nodes in B
        upper_bound = min(len(visited_a), len(matchable_b))

        return upper_bound

    def _select_diverse_seeds(
        self, candidates: List[SeedCandidate]
    ) -> List[SeedCandidate]:
        """Select a diverse set of high-quality seeds.

        Args:
            candidates: List of seed candidates.

        Returns:
            List of selected diverse seeds.
        """
        if not candidates:
            return []

        # Always include the highest-scoring seed
        selected = [candidates[0]]

        # Keep track of selected nodes to ensure diversity
        selected_a_nodes = {candidates[0].node_a}
        selected_b_nodes = {candidates[0].node_b}

        # Distance matrix to track diversity
        distances_a = self._calculate_distance_matrix(self.graph_a)
        distances_b = self._calculate_distance_matrix(self.graph_b)

        remaining_candidates = candidates[1:]

        if self.parameters.use_weighted_sampling:
            # Use weighted reservoir sampling approach
            while len(selected) < self.parameters.max_seeds and remaining_candidates:
                # Calculate diversity scores for remaining candidates
                diversity_scores = []

                for candidate in remaining_candidates:
                    # Skip if too similar to already selected seeds
                    if self._is_too_similar(
                        candidate,
                        selected_a_nodes,
                        selected_b_nodes,
                        distances_a,
                        distances_b,
                    ):
                        diversity_scores.append(0.0)
                    else:
                        # Score is product of seed quality and diversity bonus
                        diversity_scores.append(
                            candidate.score * (candidate.upper_bound or 1)
                        )

                # Normalize scores for sampling
                total_score = sum(diversity_scores)
                if total_score > 0:
                    probabilities = [score / total_score for score in diversity_scores]

                    # Sample next seed based on weights
                    idx = random.choices(
                        range(len(remaining_candidates)), weights=probabilities, k=1
                    )[0]

                    next_seed = remaining_candidates[idx]
                    selected.append(next_seed)
                    selected_a_nodes.add(next_seed.node_a)
                    selected_b_nodes.add(next_seed.node_b)

                    # Remove selected candidate
                    remaining_candidates.pop(idx)
                else:
                    # No more diverse candidates available
                    break
        else:
            # Deterministic approach
            while len(selected) < self.parameters.max_seeds and remaining_candidates:
                # Find next most diverse candidate
                best_candidate = None
                best_score = -1.0
                best_idx = -1

                for idx, candidate in enumerate(remaining_candidates):
                    # Skip if too similar to already selected seeds
                    if self._is_too_similar(
                        candidate,
                        selected_a_nodes,
                        selected_b_nodes,
                        distances_a,
                        distances_b,
                    ):
                        continue

                    # Score combines seed quality and upper bound
                    score = candidate.score * (candidate.upper_bound or 1)

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                        best_idx = idx

                if best_candidate is not None:
                    selected.append(best_candidate)
                    selected_a_nodes.add(best_candidate.node_a)
                    selected_b_nodes.add(best_candidate.node_b)
                    remaining_candidates.pop(best_idx)
                else:
                    # No more diverse candidates available
                    break

        return selected

    def _calculate_distance_matrix(
        self, graph: MolecularGraph
    ) -> Dict[Tuple[int, int], int]:
        """Calculate shortest path distances between all nodes.

        Args:
            graph: Molecular graph.

        Returns:
            Dictionary mapping node ID pairs to distances.
        """
        distances = {}
        node_ids = graph.get_node_ids()

        # Calculate all shortest paths
        for i, source in enumerate(node_ids):
            for target in node_ids[i + 1 :]:  # Avoid redundant calculations
                distance = graph.get_shortest_path_length(source, target)
                distances[(source, target)] = distance
                distances[(target, source)] = distance  # Symmetric

        return distances

    def _is_too_similar(
        self,
        candidate: SeedCandidate,
        selected_a_nodes: Set[int],
        selected_b_nodes: Set[int],
        distances_a: Dict[Tuple[int, int], int],
        distances_b: Dict[Tuple[int, int], int],
    ) -> bool:
        """Check if a candidate is too similar to already selected seeds.

        Args:
            candidate: Candidate seed to check.
            selected_a_nodes: Set of already selected nodes from graph A.
            selected_b_nodes: Set of already selected nodes from graph B.
            distances_a: Distance matrix for graph A.
            distances_b: Distance matrix for graph B.

        Returns:
            True if the candidate is too similar, False otherwise.
        """
        # Avoid selecting the same node in either graph
        if candidate.node_a in selected_a_nodes or candidate.node_b in selected_b_nodes:
            return True

        # Check topological distance to existing seeds
        for node_a in selected_a_nodes:
            distance_a = distances_a.get((candidate.node_a, node_a), float("inf"))

            # Too close in graph A
            if (
                distance_a
                < self.parameters.diversity_threshold * self.graph_a.num_nodes
            ):
                return True

        for node_b in selected_b_nodes:
            distance_b = distances_b.get((candidate.node_b, node_b), float("inf"))

            # Too close in graph B
            if (
                distance_b
                < self.parameters.diversity_threshold * self.graph_b.num_nodes
            ):
                return True

        return False
