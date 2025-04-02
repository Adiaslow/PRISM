# PRISM: Parallel Recursive Isomorphism Search for Molecules

[![PyPI version](https://badge.fury.io/py/prism-molecular.svg)](https://badge.fury.io/py/prism-molecular)
[![Documentation Status](https://readthedocs.org/projects/prism-molecular/badge/?version=latest)](https://prism-molecular.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/Adiaslow/prism/workflows/Tests/badge.svg)](https://github.com/Adiaslow/prism/actions)
[![Coverage](https://codecov.io/gh/Adiaslow/prism/branch/main/graph/badge.svg)](https://codecov.io/gh/Adiaslow/prism)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PRISM is a high-performance algorithm for finding maximum common substructures in molecular graphs. It uses a parallel recursive approach combined with advanced heuristics to efficiently identify structural similarities between molecules.

## Features

- Fast and accurate maximum common substructure (MCS) detection
- Support for multiple input formats:
  - SMILES strings
  - RDKit molecules
  - NetworkX graphs
  - Native MolecularGraph format
- Advanced node signature generation for improved matching
- Parallel processing capabilities
- Comprehensive molecular feature support
- Flexible matching parameters

## Installation

```bash
pip install prism-molecular
```

## Quick Start

```python
from prism import MolecularGraphMatcher

# Initialize the matcher
matcher = MolecularGraphMatcher()

# Find maximum common substructure between two molecules
# You can use SMILES strings directly
result = matcher.find_maximum_common_subgraph(
    "CC(=O)O",     # Acetic acid
    "CCC(=O)O"     # Propionic acid
)

# Access the results
print(f"Match size: {result.size}")
print(f"Node mapping: {result.mapping}")
print(f"Match score: {result.score}")
print(f"Time taken: {result.match_time} seconds")
```

## Advanced Usage

### Custom Parameters

```python
# Configure signature generation
signature_params = {
    "max_distance": 4,
    "use_bond_types": True,
    "use_cycles": True
}

# Configure compatibility checking
compatibility_params = {
    "element_match_required": True,
    "min_signature_similarity": 0.6,
    "progressive_refinement": True
}

# Configure matching algorithm
match_params = {
    "max_iterations": 1000,
    "timeout": 30,
    "num_threads": 4
}

# Initialize with custom parameters
matcher = MolecularGraphMatcher(
    signature_params=signature_params,
    compatibility_params=compatibility_params,
    match_params=match_params
)
```

### Using Different Input Formats

```python
# Using RDKit molecules
from rdkit import Chem
mol1 = Chem.MolFromSmiles("CC(=O)O")
mol2 = Chem.MolFromSmiles("CCC(=O)O")
result = matcher.find_maximum_common_subgraph(mol1, mol2)

# Using NetworkX graphs
import networkx as nx
graph1 = nx.Graph()
graph1.add_node(0, symbol='C')
graph1.add_node(1, symbol='C')
graph1.add_edge(0, 1, bond_type='SINGLE')
result = matcher.find_maximum_common_subgraph(graph1, graph2)
```

## Documentation

For detailed documentation, visit [prism-molecular.readthedocs.io](https://prism-molecular.readthedocs.io).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Adiaslow/prism.git
cd prism

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Citation

If you use PRISM in your research, please cite:

```bibtex
@article{prism2025,
    title={PRISM: Parallel Recursive Isomorphism Search for Molecules},
    author={Adam Murray},
    journal={:)},
    year={2025},
    volume={1},
    pages={1--10}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
