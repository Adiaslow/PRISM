"""Input handler for converting various formats to MolecularGraph."""

from typing import Union, Optional, Tuple

import networkx as nx
from rdkit import Chem

from prism.core.molecular_graph import MolecularGraph, Node, Edge


def convert_to_molecular_graph(
    input_data: Union[str, Chem.Mol, nx.Graph, MolecularGraph],
) -> MolecularGraph:
    """Convert various input formats to MolecularGraph.

    Args:
        input_data: Input in one of the following formats:
            - SMILES string
            - RDKit Mol object
            - NetworkX Graph
            - MolecularGraph

    Returns:
        MolecularGraph: Converted molecular graph

    Raises:
        ValueError: If input format is not supported or conversion fails
    """
    if isinstance(input_data, MolecularGraph):
        return input_data

    if isinstance(input_data, str):
        # Assume SMILES string
        mol = Chem.MolFromSmiles(input_data)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {input_data}")
        return MolecularGraph.from_rdkit_mol(mol)

    if isinstance(input_data, Chem.Mol):
        return MolecularGraph.from_rdkit_mol(input_data)

    if isinstance(input_data, nx.Graph):
        return _convert_networkx_to_molecular_graph(input_data)

    raise ValueError(f"Unsupported input type: {type(input_data)}")


def _convert_networkx_to_molecular_graph(graph: nx.Graph) -> MolecularGraph:
    """Convert NetworkX graph to MolecularGraph.

    Args:
        graph: NetworkX graph with node and edge attributes

    Returns:
        MolecularGraph: Converted molecular graph

    Raises:
        ValueError: If required attributes are missing
    """
    molecular_graph = MolecularGraph()

    # Convert nodes
    for node_id in graph.nodes():
        attrs = graph.nodes[node_id]

        # Required attributes
        if "symbol" not in attrs:
            raise ValueError(f"Node {node_id} missing required 'symbol' attribute")

        # Optional attributes with defaults
        features = {
            "atomic_num": attrs.get("atomic_num", 6),  # Default to carbon
            "formal_charge": attrs.get("formal_charge", 0),
            "hybridization": attrs.get("hybridization", "SP3"),
            "is_aromatic": attrs.get("is_aromatic", False),
            "num_explicit_hs": attrs.get("num_explicit_hs", 0),
            "is_in_ring": attrs.get("is_ring", False),
        }

        node = Node(
            id=node_id,
            element=attrs["symbol"],
            degree=graph.degree[node_id],
            features=features,
        )
        molecular_graph.add_node(node)

    # Convert edges
    for u, v, attrs in graph.edges(data=True):
        # Optional attributes with defaults
        features = {
            "is_conjugated": attrs.get("is_conjugated", False),
            "is_in_ring": attrs.get("is_ring", False),
            "stereo": attrs.get("stereo", "NONE"),
        }

        edge = Edge(
            source=u,
            target=v,
            bond_type=attrs.get("bond_type", "SINGLE"),
            features=features,
        )
        molecular_graph.add_edge(edge)

    return molecular_graph
