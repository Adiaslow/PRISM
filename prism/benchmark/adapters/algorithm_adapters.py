"""Algorithm adapters for benchmarking various MCS algorithms."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict, Callable

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdFMCS


def node_match(node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
    """Check if two nodes match exactly.

    Args:
        node1: First node's attributes
        node2: Second node's attributes

    Returns:
        True if nodes match exactly, False otherwise
    """
    # Must have same attributes
    if set(node1.keys()) != set(node2.keys()):
        return False

    # All attributes must match exactly
    return all(node1[key] == node2[key] for key in node1.keys())


def edge_match(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> bool:
    """Check if two edges match exactly.

    Args:
        edge1: First edge's attributes
        edge2: Second edge's attributes

    Returns:
        True if edges match exactly, False otherwise
    """
    # Must have same attributes
    if set(edge1.keys()) != set(edge2.keys()):
        return False

    # All attributes must match exactly
    return all(edge1[key] == edge2[key] for key in edge1.keys())


class NodeAttr(TypedDict):
    """Node attributes type."""

    symbol: str


class AlgorithmAdapter(ABC):
    """Base class for algorithm adapters."""

    def __init__(self, **kwargs):
        """Initialize the adapter.

        Args:
            **kwargs: Additional parameters for the algorithm
        """
        self.params = kwargs
        self._name = self.__class__.__name__
        self._description = "Base algorithm adapter"
        self.available = True

    @property
    def name(self) -> str:
        """Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        return self._name

    @property
    def description(self) -> str:
        """Get a description of the algorithm.

        Returns:
            Description of the algorithm
        """
        return self._description

    def _mol_to_networkx(self, mol: Chem.Mol) -> nx.Graph:
        """Convert RDKit molecule to NetworkX graph with exact attributes.

        Args:
            mol: RDKit molecule

        Returns:
            NetworkX graph with exact atom and bond attributes
        """
        graph = nx.Graph()

        # Add nodes with all atom attributes
        for i, atom in enumerate(mol.GetAtoms()):
            graph.add_node(
                i,
                symbol=atom.GetSymbol(),
                atomic_num=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization()),
                is_aromatic=atom.GetIsAromatic(),
                num_explicit_hs=atom.GetNumExplicitHs(),
                num_implicit_hs=atom.GetNumImplicitHs(),
                total_num_hs=atom.GetTotalNumHs(),
                is_ring=atom.IsInRing(),
            )

        # Add edges with all bond attributes
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            graph.add_edge(
                i,
                j,
                bond_type=str(bond.GetBondType()),
                is_aromatic=bond.GetIsAromatic(),
                is_conjugated=bond.GetIsConjugated(),
                is_ring=bond.IsInRing(),
            )

        return graph

    @abstractmethod
    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph between two molecules.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Dictionary containing:
                - mapping: Dictionary mapping nodes from mol1 to mol2
                - size: Size of the mapping (number of matched nodes)
                - time: Time taken in seconds
                - success: Whether the algorithm succeeded
                - error: Error message if not successful
        """
        pass


class PRISMAdapter(AlgorithmAdapter):
    """Adapter for the PRISM algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter.

        Args:
            kwargs: Algorithm-specific parameters
        """
        super().__init__(**kwargs)
        self._name = "PRISM"
        self._description = "PRISM molecular graph matching algorithm"

        try:
            from prism import MolecularGraphMatcher
            from prism.algorithm.graph_matcher import MatchParameters
            from prism.core.compatibility_matrix import CompatibilityParameters

            # Extract specific parameter groups
            signature_params = kwargs.get("signature_params", {})
            compatibility_params = kwargs.get("compatibility_params", {})
            match_params = kwargs.get("match_params", {})

            # Create parameter objects if needed
            compatibility_params_obj = None
            if compatibility_params:
                compatibility_params_obj = CompatibilityParameters(
                    **compatibility_params
                )

            match_params_obj = None
            if match_params:
                match_params_obj = MatchParameters(**match_params)

            # Initialize the algorithm
            self.matcher = MolecularGraphMatcher(
                signature_params=signature_params,
                compatibility_params=compatibility_params_obj,
                match_params=match_params_obj,
            )

        except ImportError:
            self.available = False
            self.matcher = None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using PRISM.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "PRISM not available",
            }

        try:
            # Run the algorithm and measure time
            start_time = time.time()
            result = self.matcher.find_maximum_common_subgraph(mol1, mol2)  # type: ignore
            end_time = time.time()

            return {
                "mapping": result.mapping,
                "size": result.size,
                "time": end_time - start_time,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class RDKitMCSAdapter(AlgorithmAdapter):
    """Adapter for RDKit's MCS algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "RDKit-MCS"
        self._description = (
            "RDKit's maximum common subgraph algorithm (NetworkX implementation)"
        )

        # RDKit MCS parameters
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = nx is not None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using NetworkX's implementation.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "NetworkX not available",
            }

        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using NetworkX
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class McGregorAdapter(AlgorithmAdapter):
    """Adapter for McGregor's algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "McGregor"
        self._description = "McGregor's maximum common subgraph algorithm"

        # McGregor parameters for exact matching
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = True

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using McGregor's algorithm.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using NetworkX
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class MCSPlusAdapter(AlgorithmAdapter):
    """Adapter for MCSP+ algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "MCSP+"
        self._description = (
            "MCSP+ maximum common subgraph algorithm (NetworkX implementation)"
        )

        # MCSP+ parameters
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = nx is not None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using NetworkX's implementation.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "NetworkX not available",
            }

        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using NetworkX
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class UllmannAdapter(AlgorithmAdapter):
    """Adapter for Ullmann's algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "Ullmann"
        self._description = "Ullmann's maximum common subgraph algorithm"

        # Ullmann parameters
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = nx is not None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using NetworkX's implementation.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "NetworkX not available",
            }

        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using NetworkX
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class CliquePlusAdapter(AlgorithmAdapter):
    """Adapter for Clique+ algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "Clique+"
        self._description = "Clique+ maximum common subgraph algorithm"

        # Clique+ parameters
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = nx is not None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using NetworkX's implementation.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "NetworkX not available",
            }

        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using NetworkX
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


class VF2Adapter(AlgorithmAdapter):
    """Adapter for NetworkX's VF2 algorithm."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        super().__init__(**kwargs)
        self._name = "VF2"
        self._description = "NetworkX's VF2 maximum common subgraph algorithm"

        # VF2 parameters
        self.params = {
            "timeout": kwargs.get("timeout", 60),  # seconds
            "min_size": kwargs.get("min_size", 3),
        }

        self.available = nx is not None

    def find_maximum_common_subgraph(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Find the maximum common subgraph using VF2.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Result dictionary
        """
        if not self.available:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": "NetworkX not available",
            }

        try:
            # Convert molecules to NetworkX graphs with exact attributes
            graph1 = self._mol_to_networkx(mol1)
            graph2 = self._mol_to_networkx(mol2)

            # Run the algorithm and measure time
            start_time = time.time()

            # Find maximum common subgraph using VF2
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                graph2,
                graph1,
                node_match=node_match,
                edge_match=edge_match,
            )

            # Get the largest common subgraph
            best_mapping = {}
            best_size = 0

            for mapping in matcher.subgraph_isomorphisms_iter():
                if len(mapping) > best_size:
                    best_mapping = mapping
                    best_size = len(mapping)

            end_time = time.time()

            # Convert mapping to the expected format
            result_mapping = {v: k for k, v in best_mapping.items()}

            return {
                "mapping": result_mapping,
                "size": best_size,
                "time": end_time - start_time,
                "success": best_size >= self.params["min_size"],
                "error": None,
            }

        except Exception as e:
            return {
                "mapping": {},
                "size": 0,
                "time": 0.0,
                "success": False,
                "error": str(e),
            }


# Factory function to get all available algorithm adapters
def get_all_algorithm_adapters(**kwargs) -> Dict[str, AlgorithmAdapter]:
    """Get all available algorithm adapters.

    Args:
        kwargs: Algorithm-specific parameters

    Returns:
        Dictionary of algorithm name to adapter
    """
    adapters = {
        "PRISM": PRISMAdapter(**kwargs.get("prism_params", {})),
        "RDKit-MCS": RDKitMCSAdapter(**kwargs.get("rdkit_params", {})),
        "VF2": VF2Adapter(**kwargs.get("vf2_params", {})),
        "McGregor": McGregorAdapter(**kwargs.get("mcgregor_params", {})),
        "MCSP+": MCSPlusAdapter(**kwargs.get("mcsp_params", {})),
        "Ullmann": UllmannAdapter(**kwargs.get("ullmann_params", {})),
        "Clique+": CliquePlusAdapter(**kwargs.get("clique_params", {})),
    }

    return {name: adapter for name, adapter in adapters.items() if adapter.available}
