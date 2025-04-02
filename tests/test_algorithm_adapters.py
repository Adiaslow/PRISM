"""Test module for algorithm adapters.

This module contains tests for all algorithm adapters to ensure they work correctly
with molecular graph matching.
"""

import unittest

from rdkit import Chem

from prism.benchmark.adapters.algorithm_adapters import (
    CliquePlusAdapter,
    McGregorAdapter,
    MCSPlusAdapter,
    PRISMAdapter,
    RDKitMCSAdapter,
    UllmannAdapter,
    VF2Adapter,
    get_all_algorithm_adapters,
)


class TestAlgorithmAdapters(unittest.TestCase):
    """Test class for algorithm adapters."""

    def setUp(self):
        """Set up test environment."""
        # Create test molecules
        self.mol1 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        self.mol2 = Chem.MolFromSmiles("c1ccccc1C")  # Toluene
        self.mol3 = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
        self.mol4 = Chem.MolFromSmiles("CC(=O)OC")  # Methyl acetate

    def test_prism_adapter(self):
        """Test PRISM adapter."""
        adapter = PRISMAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_rdkit_mcs_adapter(self):
        """Test RDKit MCS adapter."""
        adapter = RDKitMCSAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_vf2_adapter(self):
        """Test VF2 adapter."""
        adapter = VF2Adapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_mcgregor_adapter(self):
        """Test McGregor adapter."""
        adapter = McGregorAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_mcsplus_adapter(self):
        """Test MCSP+ adapter."""
        adapter = MCSPlusAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_ullmann_adapter(self):
        """Test Ullmann adapter."""
        adapter = UllmannAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_cliqueplus_adapter(self):
        """Test Clique+ adapter."""
        adapter = CliquePlusAdapter()
        result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
        self.assertTrue(result["success"])
        self.assertGreater(result["size"], 0)
        self.assertGreater(result["time"], 0)

    def test_get_all_algorithm_adapters(self):
        """Test getting all algorithm adapters."""
        adapters = get_all_algorithm_adapters()
        self.assertIsInstance(adapters, dict)
        self.assertGreater(len(adapters), 0)
        for name, adapter in adapters.items():
            self.assertTrue(hasattr(adapter, "find_maximum_common_subgraph"))

    def test_algorithm_comparison(self):
        """Test that different algorithms produce similar results."""
        adapters = get_all_algorithm_adapters()
        results = {}

        for name, adapter in adapters.items():
            result = adapter.find_maximum_common_subgraph(self.mol1, self.mol2)
            results[name] = result

        # Check that all algorithms found a solution
        for name, result in results.items():
            self.assertTrue(result["success"], f"{name} failed to find a solution")

        # Check that solution sizes are similar
        sizes = [result["size"] for result in results.values()]
        max_size_diff = max(sizes) - min(sizes)
        self.assertLess(max_size_diff, 3)  # Allow small differences in solution size


if __name__ == "__main__":
    unittest.main()
