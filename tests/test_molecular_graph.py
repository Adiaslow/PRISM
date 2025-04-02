"""Unit tests for MolecularGraph class."""

import unittest

import networkx as nx
import numpy as np
from rdkit import Chem

from prism.core.molecular_graph import MolecularGraph, Node, Edge


class TestMolecularGraph(unittest.TestCase):
    """Test suite for MolecularGraph class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple molecule (benzene)
        self.benzene = Chem.MolFromSmiles("c1ccccc1")
        self.benzene_graph = MolecularGraph.from_rdkit_mol(self.benzene)

        # Create another molecule (pyridine)
        self.pyridine = Chem.MolFromSmiles("c1ccncc1")
        self.pyridine_graph = MolecularGraph.from_rdkit_mol(self.pyridine)

    def test_create_from_rdkit(self):
        """Test creation from RDKit molecules."""
        self.assertEqual(self.benzene_graph.num_nodes, 6)
        self.assertEqual(self.benzene_graph.num_edges, 6)

        self.assertEqual(self.pyridine_graph.num_nodes, 6)
        self.assertEqual(self.pyridine_graph.num_edges, 6)

        # Check element types
        for i in range(6):
            node = self.benzene_graph.get_node(i)
            self.assertIsNotNone(node)
            self.assertEqual(node.element, "C")

        # Check pyridine has nitrogen
        nitrogen_found = False
        for i in range(6):
            node = self.pyridine_graph.get_node(i)
            self.assertIsNotNone(node)
            if node.element == "N":
                nitrogen_found = True
        self.assertTrue(nitrogen_found)

    def test_adjacency(self):
        """Test adjacency information."""
        # Check benzene connectivity
        for i in range(6):
            neighbors = self.benzene_graph.get_adjacent_nodes(i)
            # Each carbon in benzene has 2 neighbors
            self.assertEqual(len(neighbors), 2)

            # Check connectivity: node i should be connected to (i-1)%6 and (i+1)%6
            self.assertIn((i + 1) % 6, neighbors)
            self.assertIn((i - 1) % 6, neighbors)

    def test_edge_access(self):
        """Test edge access."""
        # Check benzene bond types
        for i in range(6):
            next_idx = (i + 1) % 6
            edge = self.benzene_graph.get_edge_between(i, next_idx)

            self.assertIsNotNone(edge)
            self.assertEqual(edge.bond_type, "AROMATIC")

            # Test features
            self.assertTrue(edge.features["is_in_ring"])
            self.assertTrue(edge.features["is_conjugated"])

    def test_node_signatures(self):
        """Test node signature setting and getting."""
        # Create test signatures
        for i in range(6):
            sig = np.array([i, i + 1, i + 2], dtype=float)
            self.benzene_graph.set_node_signature(i, sig)

        # Check retrieval
        for i in range(6):
            sig = self.benzene_graph.get_node_signature(i)
            self.assertIsNotNone(sig)
            self.assertEqual(sig[0], i)
            self.assertEqual(sig[1], i + 1)
            self.assertEqual(sig[2], i + 2)

    def test_shortest_path(self):
        """Test shortest path calculation."""
        # In benzene, the maximum shortest path length should be 3
        max_path_len = 0

        for i in range(6):
            for j in range(i + 1, 6):
                path = self.benzene_graph.get_shortest_path(i, j)
                max_path_len = max(
                    max_path_len, len(path) - 1
                )  # Path includes endpoints

                # Check path length
                path_len = self.benzene_graph.get_shortest_path_length(i, j)
                self.assertEqual(path_len, len(path) - 1)

        self.assertEqual(max_path_len, 3)

    def test_subgraph(self):
        """Test subgraph creation."""
        # Create a subgraph with nodes 0, 1, 2
        subgraph = self.benzene_graph.get_subgraph([0, 1, 2])

        self.assertEqual(subgraph.num_nodes, 3)
        # Nodes 0-1, 1-2 should be connected
        self.assertEqual(subgraph.num_edges, 2)

        # Check connectivity
        self.assertIn(1, subgraph.get_adjacent_nodes(0))
        self.assertIn(0, subgraph.get_adjacent_nodes(1))
        self.assertIn(2, subgraph.get_adjacent_nodes(1))
        self.assertIn(1, subgraph.get_adjacent_nodes(2))

        # Nodes 0 and 2 should not be connected
        self.assertNotIn(2, subgraph.get_adjacent_nodes(0))
        self.assertNotIn(0, subgraph.get_adjacent_nodes(2))

    def test_cycles(self):
        """Test cycle detection."""
        # Benzene should have one cycle containing all nodes
        for i in range(6):
            cycles = self.benzene_graph.get_node_cycles(i)
            self.assertEqual(len(cycles), 1)
            self.assertEqual(len(cycles[0]), 6)

        # Create a non-cyclic molecule
        propane = Chem.MolFromSmiles("CCC")
        propane_graph = MolecularGraph.from_rdkit_mol(propane)

        for i in range(3):
            cycles = propane_graph.get_node_cycles(i)
            self.assertEqual(len(cycles), 0)


if __name__ == "__main__":
    unittest.main()
