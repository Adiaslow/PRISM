# prism/algorithm/graph_matcher.py

"""Bidirectional A* guided graph matching algorithm."""

import heapq
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from prism.core.molecular_graph import MolecularGraph
from prism.core.compatibility_matrix import CompatibilityMatrix
from prism.algorithm.seed_selection import SeedCandidate, SeedSelector


class MatchResult(BaseModel):
    """Result of a graph matching operation."""

    mapping: Dict[int, int] = Field(default_factory=dict)
    size: int = 0
    score: float = 0.0
    match_time: float = 0.0

    class Config:
        """Pydantic model configuration."""

        frozen = True


class MatchParameters(BaseModel):
    """Parameters for controlling the matching process."""

    max_time_seconds: int = 60
    max_iterations: int = 1000000
    num_threads: int = 4
    use_forward_checking: bool = True
    use_conflict_backjumping: bool = True

    class Config:
        """Pydantic model configuration."""

        frozen = True


@total_ordering
@dataclass
class PartialMatch:
    """Represents a partial match during the search process."""

    # Current mapping from graph A nodes to graph B nodes
    mapping: Dict[int, int] = field(default_factory=dict)

    # Nodes that have been mapped
    mapped_a: Set[int] = field(default_factory=set)
    mapped_b: Set[int] = field(default_factory=set)

    # Frontier nodes to consider for expansion
    frontier_a: Set[int] = field(default_factory=set)
    frontier_b: Set[int] = field(default_factory=set)

    # Score components
    current_size: int = 0
    current_score: float = 0.0
    estimated_future_score: float = 0.0

    # Conflict tracking for backjumping
    conflict_set: Dict[int, Set[int]] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare based on total estimated score (for priority queue)."""
        if not isinstance(other, PartialMatch):
            return NotImplemented
        return (self.current_score + self.estimated_future_score) > (
            other.current_score + other.estimated_future_score
        )

    def __eq__(self, other):
        """Check equality based on mappings."""
        if not isinstance(other, PartialMatch):
            return NotImplemented
        return self.mapping == other.mapping

    def copy(self):
        """Create a deep copy of this partial match."""
        return PartialMatch(
            mapping=self.mapping.copy(),
            mapped_a=self.mapped_a.copy(),
            mapped_b=self.mapped_b.copy(),
            frontier_a=self.frontier_a.copy(),
            frontier_b=self.frontier_b.copy(),
            current_size=self.current_size,
            current_score=self.current_score,
            estimated_future_score=self.estimated_future_score,
            conflict_set={k: v.copy() for k, v in self.conflict_set.items()},
        )


class GraphMatcher:
    """Implements the bidirectional A* guided graph matching algorithm."""

    def __init__(
        self,
        graph_a: MolecularGraph,
        graph_b: MolecularGraph,
        compatibility_matrix: CompatibilityMatrix,
        parameters: Optional[MatchParameters] = None,
    ):
        """Initialize the graph matcher.

        Args:
            graph_a: First molecular graph.
            graph_b: Second molecular graph.
            compatibility_matrix: Compatibility matrix between the graphs.
            parameters: Optional matching parameters.
        """
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.compatibility_matrix = compatibility_matrix
        self.parameters = parameters or MatchParameters()

        # Results shared across threads
        self.best_match = MatchResult()
        self.match_lock = threading.Lock()

        # Termination flags
        self.terminate = False
        self.start_time = 0.0

    def find_maximum_common_subgraph(
        self, seeds: Optional[List[SeedCandidate]] = None
    ) -> MatchResult:
        """Find the maximum common subgraph between the two graphs.

        Args:
            seeds: Optional list of seed node pairs to start from.

        Returns:
            Result containing the mapping and statistics.
        """
        if seeds is None or not seeds:
            # Generate seeds automatically if not provided
            seed_selector = SeedSelector(
                self.graph_a, self.graph_b, self.compatibility_matrix
            )
            seeds = seed_selector.select_seeds()

        if not seeds:
            # No valid seeds found
            return MatchResult()

        # Record start time
        self.start_time = time.time()
        self.best_match = MatchResult()
        self.terminate = False

        # Launch parallel search from different seeds
        with ThreadPoolExecutor(
            max_workers=min(self.parameters.num_threads, len(seeds))
        ) as executor:
            # Submit each seed for exploration
            futures = [executor.submit(self._explore_from_seed, seed) for seed in seeds]

            # Wait for all threads to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread: {e}")

        # Record search time
        search_time = time.time() - self.start_time

        # Update final result with timing
        with self.match_lock:
            result = MatchResult(
                mapping=self.best_match.mapping,
                size=self.best_match.size,
                score=self.best_match.score,
                match_time=search_time,
            )

        return result

    def _explore_from_seed(self, seed: SeedCandidate) -> None:
        """Explore the search space starting from a seed pair.

        Args:
            seed: Seed pair to start from.
        """
        if self.terminate:
            return

        # Initialize partial match with the seed pair
        initial_match = PartialMatch()
        initial_match.mapping = {seed.node_a: seed.node_b}
        initial_match.mapped_a = {seed.node_a}
        initial_match.mapped_b = {seed.node_b}
        initial_match.current_size = 1
        initial_match.current_score = self.compatibility_matrix.get_compatibility(
            seed.node_a, seed.node_b
        )

        # Initialize frontiers with adjacent nodes
        self._update_frontiers(initial_match)

        # Estimate future score
        initial_match.estimated_future_score = self._estimate_future_score(
            initial_match
        )

        # Initialize priority queue with initial match
        queue = [initial_match]

        # A* search
        iterations = 0

        while queue and not self.terminate:
            # Check termination conditions
            iterations += 1
            current_time = time.time()

            if (
                iterations >= self.parameters.max_iterations
                or current_time - self.start_time >= self.parameters.max_time_seconds
            ):
                break

            # Get next partial match from queue
            current_match = heapq.heappop(queue)

            # If this match is already worse than the best match, discard it
            with self.match_lock:
                if current_match.current_size < self.best_match.size:
                    continue

            # Check if this is a complete match (no more nodes to add)
            if not current_match.frontier_a or not current_match.frontier_b:
                # Update best match if this one is better
                with self.match_lock:
                    if current_match.current_size > self.best_match.size or (
                        current_match.current_size == self.best_match.size
                        and current_match.current_score > self.best_match.score
                    ):
                        self.best_match = MatchResult(
                            mapping=current_match.mapping.copy(),
                            size=current_match.current_size,
                            score=current_match.current_score,
                            match_time=time.time() - self.start_time,
                        )
                continue

            # Find best node pair to add to the match
            candidates = self._find_expansion_candidates(current_match)

            for node_a, node_b, score in candidates:
                # Skip if match is invalid
                if node_a in current_match.mapped_a or node_b in current_match.mapped_b:
                    continue

                # Create new match with this pair added
                new_match = current_match.copy()
                new_match.mapping[node_a] = node_b
                new_match.mapped_a.add(node_a)
                new_match.mapped_b.add(node_b)
                new_match.current_size += 1
                new_match.current_score += score

                # Remove the added nodes from frontiers
                new_match.frontier_a.discard(node_a)
                new_match.frontier_b.discard(node_b)

                # Update frontiers with new adjacent nodes
                self._update_frontiers(new_match)

                # Check consistency if using forward checking
                if self.parameters.use_forward_checking:
                    is_consistent, conflicts = self._check_forward_consistency(
                        new_match
                    )
                    if not is_consistent:
                        if self.parameters.use_conflict_backjumping:
                            # Store conflict information for backjumping
                            for conflict_node in conflicts:
                                if conflict_node not in new_match.conflict_set:
                                    new_match.conflict_set[conflict_node] = set()
                                new_match.conflict_set[conflict_node].add(node_a)
                        continue

                # Update future score estimate
                new_match.estimated_future_score = self._estimate_future_score(
                    new_match
                )

                # Add to priority queue
                heapq.heappush(queue, new_match)

    def _update_frontiers(self, match: PartialMatch) -> None:
        """Update the frontier sets with adjacent unmapped nodes.

        Args:
            match: Partial match to update.
        """
        # Find all adjacent nodes to the current mapping
        new_frontier_a = set()
        new_frontier_b = set()

        for node_a in match.mapped_a:
            for adj in self.graph_a.get_adjacent_nodes(node_a):
                if adj not in match.mapped_a:
                    new_frontier_a.add(adj)

        for node_b in match.mapped_b:
            for adj in self.graph_b.get_adjacent_nodes(node_b):
                if adj not in match.mapped_b:
                    new_frontier_b.add(adj)

        # Update frontiers
        match.frontier_a.update(new_frontier_a)
        match.frontier_b.update(new_frontier_b)

    def _find_expansion_candidates(
        self, match: PartialMatch
    ) -> List[Tuple[int, int, float]]:
        """Find best candidate pairs for expanding the current match.

        Args:
            match: Current partial match.

        Returns:
            List of (node_a, node_b, score) tuples for expansion.
        """
        candidates = []

        # Score all possible pairs in the frontier
        for node_a in match.frontier_a:
            for node_b in match.frontier_b:
                # Skip incompatible pairs
                compatibility = self.compatibility_matrix.get_compatibility(
                    node_a, node_b
                )
                if compatibility <= 0:
                    continue

                # Check topological consistency with existing mapping
                if not self._check_topological_consistency(match, node_a, node_b):
                    continue

                # Calculate multi-factor score
                score = self._calculate_expansion_score(
                    match, node_a, node_b, compatibility
                )

                # Add to candidates
                candidates.append((node_a, node_b, score))

        # Sort by score (descending) and return top candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:10]  # Limit to top 10 candidates for efficiency

    def _check_topological_consistency(
        self, match: PartialMatch, node_a: int, node_b: int
    ) -> bool:
        """Check if adding a pair maintains topological consistency.

        Args:
            match: Current partial match.
            node_a: Node from graph A to add.
            node_b: Node from graph B to add.

        Returns:
            True if topologically consistent, False otherwise.
        """
        # Check edge consistency
        for existing_a, existing_b in match.mapping.items():
            # Check if edge exists in one graph but not the other
            edge_a_exists = existing_a in self.graph_a.get_adjacent_nodes(node_a)
            edge_b_exists = existing_b in self.graph_b.get_adjacent_nodes(node_b)

            if edge_a_exists != edge_b_exists:
                return False

            # If edges exist in both graphs, check if bond types are compatible
            if edge_a_exists and edge_b_exists:
                edge_a = self.graph_a.get_edge_between(node_a, existing_a)
                edge_b = self.graph_b.get_edge_between(node_b, existing_b)

                if edge_a is None or edge_b is None:
                    continue

                # Check if bond types are compatible (can be extended for more complex rules)
                # For example, single bonds might be allowed to match double bonds in some contexts
                if edge_a.bond_type != edge_b.bond_type:
                    # Allow aromatic to match aromatic or double bonds
                    if not (
                        (
                            edge_a.bond_type == "AROMATIC"
                            and edge_b.bond_type in ["AROMATIC", "DOUBLE"]
                        )
                        or (
                            edge_b.bond_type == "AROMATIC"
                            and edge_a.bond_type in ["AROMATIC", "DOUBLE"]
                        )
                    ):
                        return False

        return True

    def _check_forward_consistency(self, match: PartialMatch) -> Tuple[bool, Set[int]]:
        """Check if the current match can lead to a valid complete matching.

        Args:
            match: Current partial match.

        Returns:
            Tuple of (is_consistent, conflict_nodes).
        """
        # Check if each unmapped node in frontier_a has at least one compatible node in frontier_b
        conflicts = set()

        for node_a in match.frontier_a:
            has_compatible = False

            for node_b in match.frontier_b:
                if self.compatibility_matrix.get_compatibility(
                    node_a, node_b
                ) > 0 and self._check_topological_consistency(match, node_a, node_b):
                    has_compatible = True
                    break

            if not has_compatible:
                conflicts.add(node_a)

        # If any node has no compatible matches, the matching is inconsistent
        return len(conflicts) == 0, conflicts

    def _calculate_expansion_score(
        self, match: PartialMatch, node_a: int, node_b: int, compatibility: float
    ) -> float:
        """Calculate multi-factor score for a potential expansion.

        Args:
            match: Current partial match.
            node_a: Node from graph A to add.
            node_b: Node from graph B to add.
            compatibility: Base compatibility score.

        Returns:
            Combined expansion score.
        """
        # Current contribution to overall quality
        current_contribution = compatibility

        # Structural consistency with existing match
        structural_consistency = 0.0
        connected_to_existing = 0

        for existing_a, existing_b in match.mapping.items():
            # Check if connected to existing nodes
            if existing_a in self.graph_a.get_adjacent_nodes(node_a):
                connected_to_existing += 1

                # Check edge similarity
                edge_a = self.graph_a.get_edge_between(node_a, existing_a)
                edge_b = self.graph_b.get_edge_between(node_b, existing_b)

                if edge_a is not None and edge_b is not None:
                    # Bond type similarity (1.0 if identical, 0.5 if compatible, 0.0 otherwise)
                    if edge_a.bond_type == edge_b.bond_type:
                        structural_consistency += 1.0
                    elif (
                        edge_a.bond_type == "AROMATIC"
                        and edge_b.bond_type in ["AROMATIC", "DOUBLE"]
                    ) or (
                        edge_b.bond_type == "AROMATIC"
                        and edge_a.bond_type in ["AROMATIC", "DOUBLE"]
                    ):
                        structural_consistency += 0.5

        # Normalize structural consistency
        if connected_to_existing > 0:
            structural_consistency /= connected_to_existing

        # Future expansion potential
        expansion_potential = 0.0

        # Count unmatched neighbors
        unmatched_neighbors_a = len(
            [
                n
                for n in self.graph_a.get_adjacent_nodes(node_a)
                if n not in match.mapped_a
            ]
        )

        unmatched_neighbors_b = len(
            [
                n
                for n in self.graph_b.get_adjacent_nodes(node_b)
                if n not in match.mapped_b
            ]
        )

        # Prefer nodes with similar numbers of unmatched neighbors
        max_unmatched = max(unmatched_neighbors_a, unmatched_neighbors_b)
        if max_unmatched > 0:
            expansion_potential = (
                1.0 - abs(unmatched_neighbors_a - unmatched_neighbors_b) / max_unmatched
            )
        else:
            expansion_potential = 1.0

        # Global similarity impact (reward nodes likely to lead to larger matches)
        global_impact = (
            min(unmatched_neighbors_a, unmatched_neighbors_b) / 10.0
        )  # Scale factor

        # Combine all factors with weights
        weights = [0.3, 0.3, 0.2, 0.2]  # Current, structural, expansion, global
        total_score = (
            weights[0] * current_contribution
            + weights[1] * structural_consistency
            + weights[2] * expansion_potential
            + weights[3] * global_impact
        )

        return total_score

    def _estimate_future_score(self, match: PartialMatch) -> float:
        """Estimate future reward for A* heuristic.

        Args:
            match: Current partial match.

        Returns:
            Estimated future score.
        """
        # Simple greedy estimate: find how many more pairs could potentially be added
        potential_additions = 0

        # For each frontier node in A, count compatible frontier nodes in B
        for node_a in match.frontier_a:
            compatible_count = 0

            for node_b in match.frontier_b:
                if self.compatibility_matrix.get_compatibility(node_a, node_b) > 0:
                    compatible_count += 1

            # Each node in A can match at most one node in B
            potential_additions += min(1, compatible_count)

        # Optimistic future score estimate
        return potential_additions * 0.5  # Average expected score per future match
