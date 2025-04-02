"""Unit tests for input handler functionality."""

import unittest

import networkx as nx
from rdkit import Chem

from prism.core.molecular_graph import MolecularGraph
from prism.utils.input_handler import convert_to_molecular_graph


class TestInputHandler(unittest.TestCase):
    """Test suite for input handler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Test SMILES string
        self.smiles = "CC(=O)O"  # Acetic acid

        # Test RDKit molecule
        self.rdkit_mol = Chem.MolFromSmiles(self.smiles)

        # Test NetworkX graph
        self.nx_graph = nx.Graph()
        # Add nodes
        self.nx_graph.add_node(0, symbol="C")
        self.nx_graph.add_node(1, symbol="C")
        self.nx_graph.add_node(2, symbol="O")
        self.nx_graph.add_node(3, symbol="O")
        # Add edges
        self.nx_graph.add_edge(0, 1, bond_type="SINGLE")
        self.nx_graph.add_edge(1, 2, bond_type="DOUBLE")
        self.nx_graph.add_edge(1, 3, bond_type="SINGLE")

        # Test MolecularGraph
        self.mol_graph = MolecularGraph.from_rdkit_mol(self.rdkit_mol)

    def test_smiles_conversion(self):
        """Test conversion from SMILES string."""
        graph = convert_to_molecular_graph(self.smiles)
        self.assertIsInstance(graph, MolecularGraph)
        self.assertEqual(graph.num_nodes, 4)  # C, C, O, O
        self.assertEqual(graph.num_edges, 3)  # C-C, C=O, C-O

    def test_rdkit_conversion(self):
        """Test conversion from RDKit molecule."""
        graph = convert_to_molecular_graph(self.rdkit_mol)
        self.assertIsInstance(graph, MolecularGraph)
        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.num_edges, 3)

    def test_networkx_conversion(self):
        """Test conversion from NetworkX graph."""
        graph = convert_to_molecular_graph(self.nx_graph)
        self.assertIsInstance(graph, MolecularGraph)
        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.num_edges, 3)

        # Check node attributes
        for node_id in range(4):
            node = graph.get_node(node_id)
            self.assertIsNotNone(node)
            self.assertEqual(node.element, self.nx_graph.nodes[node_id]["symbol"])

    def test_molecular_graph_passthrough(self):
        """Test that MolecularGraph input is passed through unchanged."""
        graph = convert_to_molecular_graph(self.mol_graph)
        self.assertIs(graph, self.mol_graph)  # Should be the same object

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES string."""
        with self.assertRaises(ValueError):
            convert_to_molecular_graph("invalid_smiles")

    def test_invalid_networkx(self):
        """Test handling of invalid NetworkX graph."""
        # Graph missing required 'symbol' attribute
        invalid_graph = nx.Graph()
        invalid_graph.add_node(0)
        with self.assertRaises(ValueError):
            convert_to_molecular_graph(invalid_graph)

    def test_unsupported_type(self):
        """Test handling of unsupported input type."""
        with self.assertRaises(ValueError):
            convert_to_molecular_graph(42)  # Integer is not a supported type
