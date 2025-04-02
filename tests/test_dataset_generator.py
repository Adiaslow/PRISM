"""Test module for the benchmark dataset generator.

This module contains tests for the dataset generator functionality
to ensure it creates valid molecular pairs with diverse atom types.
"""

import unittest
from collections import Counter

from rdkit import Chem

from prism.benchmark.configs.benchmark_config import BenchmarkConfig
from prism.benchmark.services.dataset_generator import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):
    """Test class for the DatasetGenerator."""

    def setUp(self):
        """Set up test environment."""
        self.config = BenchmarkConfig.default_config()
        self.generator = DatasetGenerator(self.config)
        # Load the dataset
        dataset_path = "benchmark_data/pubchem_metabolite_pathways_filter.csv"
        loaded_count = self.generator.load_molecular_dataset(dataset_path)
        self.assertTrue(loaded_count > 0, "Failed to load molecules from dataset")

    def test_generate_dataset(self):
        """Test generating a small dataset."""
        dataset = self.generator.generate_dataset()

        # Check if dataset was created
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset.pairs) > 0)

        # Check if each pair has valid molecules
        for pair in dataset.pairs[:5]:  # Test first 5 pairs
            self.assertIsNotNone(pair.smiles1)
            self.assertIsNotNone(pair.smiles2)

            mol1 = Chem.MolFromSmiles(pair.smiles1)
            mol2 = Chem.MolFromSmiles(pair.smiles2)

            self.assertIsNotNone(mol1)
            self.assertIsNotNone(mol2)

            # Check sizes
            self.assertEqual(pair.mol1_size, mol1.GetNumAtoms())
            self.assertEqual(pair.mol2_size, mol2.GetNumAtoms())

    def test_atom_diversity(self):
        """Test that the dataset contains diverse atom types."""
        dataset = self.generator.generate_dataset()

        # Collect all atoms
        all_atoms = []
        for pair in dataset.pairs:
            mol1 = Chem.MolFromSmiles(pair.smiles1)
            mol2 = Chem.MolFromSmiles(pair.smiles2)

            if mol1:
                all_atoms.extend([atom.GetSymbol() for atom in mol1.GetAtoms()])
            if mol2:
                all_atoms.extend([atom.GetSymbol() for atom in mol2.GetAtoms()])

        # Count atoms
        atom_counts = Counter(all_atoms)
        print(f"Atom distribution: {dict(atom_counts)}")  # Add debug output

        # Check that we have at least these common atoms
        expected_atoms = ["C", "O", "N"]
        for atom in expected_atoms:
            self.assertIn(atom, atom_counts)

        # Check for biological diversity - at least 3 different atom types
        self.assertGreaterEqual(len(atom_counts), 3)


if __name__ == "__main__":
    unittest.main()
