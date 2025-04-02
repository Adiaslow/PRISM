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

## Algorithm Overview

PRISM combines several innovative techniques to efficiently solve the molecular subgraph isomorphism problem, which is NP-hard in general. Here's a conceptual breakdown of its key components and optimizations:

### Core Components

1. **Molecular Fingerprinting**

   - Advanced node signature generation that captures both local and global structural information
   - Multi-scale topological features incorporating:
     - Atomic properties (element type, charge, hybridization)
     - Local connectivity patterns
     - Extended neighborhood information
     - Ring system participation
   - Efficient vector representations enabling fast similarity computations

2. **Intelligent Search Space Reduction**

   - Progressive compatibility matrix refinement
   - Early pruning of incompatible node pairs
   - Dynamic filtering based on structural constraints
   - Neighborhood-aware compatibility scoring

3. **Strategic Seed Selection**

   - Multi-criteria ranking system for initial match points
   - Balance between:
     - Local structural uniqueness
     - Global topological importance
     - Potential for match expansion
     - Connectivity patterns
   - Diversity-based sampling to avoid local optima

4. **Bidirectional Search Strategy**
   - Simultaneous forward and backward search
   - A\* guidance with admissible heuristics
   - Dynamic priority queue management
   - Efficient partial solution evaluation

### Key Optimizations

1. **Parallel Processing**

   - Multi-threaded search execution
   - Independent subproblem exploration
   - Load-balanced work distribution
   - Thread-safe data structures

2. **Memory Management**

   - Compact signature representations
   - Sparse matrix operations
   - Efficient backtracking mechanisms
   - Memory-conscious data structures

3. **Performance Enhancements**

   - Cached intermediate results
   - Pre-computed structural features
   - Fast compatibility checks
   - Optimized vector operations

4. **Search Space Optimization**
   - Dynamic branching factor reduction
   - Incremental consistency checking
   - Adaptive search depth control
   - Smart backtracking strategies

### Algorithm Workflow

1. **Preprocessing Phase**

   - Generate molecular graph representations
   - Compute node signatures
   - Build initial compatibility matrix
   - Identify potential seed pairs

2. **Search Phase**

   - Select and prioritize seed pairs
   - Launch parallel search threads
   - Perform bidirectional exploration
   - Update global solution state

3. **Solution Refinement**
   - Validate partial solutions
   - Merge parallel results
   - Optimize final mapping
   - Verify solution consistency

### Mathematical Formulation

Let $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ be two molecular graphs, where:

- $V_i$ is the set of nodes (atoms) in graph $i$
- $E_i$ is the set of edges (bonds) in graph $i$

#### 1. Node Signature Generation

For each node $v$ in a molecular graph, we compute a signature vector $\sigma(v)$ that combines:

$$\sigma(v) = w_e\phi_e(v) + w_c\phi_c(v) + w_n\phi_n(v) + w_r\phi_r(v)$$

where:

- $\phi_e(v)$: Element-type encoding (one-hot vector)
- $\phi_c(v)$: Connectivity pattern (degree and bond types)
- $\phi_n(v)$: Neighborhood topology up to distance $d$
- $\phi_r(v)$: Ring participation vector
- $w_e, w_c, w_n, w_r \in \mathbb{R}^+$: Non-negative weight parameters

The neighborhood topology component is defined as:

$$\phi_n(v) = \sum_{i=1}^d \alpha^i \sum_{u \in N_i(v)} \phi_e(u)$$

where:

- $d \in \mathbb{N}$: Maximum neighborhood distance considered
- $\alpha \in (0, 1)$: Decay factor for distance weighting
- $N_i(v)$: Set of nodes at exact distance $i$ from node $v$

#### 2. Node Compatibility

For nodes $v_1 \in V_1, v_2 \in V_2$ from different molecules, compatibility is computed as:

$$C(v_1, v_2) = \frac{\sigma(v_1) \cdot \sigma(v_2)}{\|\sigma(v_1)\| \|\sigma(v_2)\|} \cdot \delta(v_1, v_2)$$

where $\delta(v_1, v_2)$ is a binary constraint satisfaction term:

$$
\delta(v_1, v_2) = \begin{cases}
1 & \text{if all constraints are satisfied} \\
0 & \text{otherwise}
\end{cases}
$$

Constraints include:

- Element type compatibility
- Degree compatibility:

$$|deg(v_1) - deg(v_2)| \leq \tau_{deg}$$

- Bond type compatibility
- Valence compatibility

where:

- $deg(v)$: Degree (number of bonds) of node $v$
- $\tau_{deg} \in \mathbb{N}$: Degree difference threshold

#### 3. Seed Selection

Seed pairs are ranked using:

$$S(v_1, v_2) = w_u U(v_1, v_2) + w_s I(v_1, v_2) + w_e E(v_1, v_2) + w_c K(v_1, v_2)$$

where:

- $w_u, w_s, w_e, w_c \in \mathbb{R}^+$: Non-negative weight parameters
- $U(v_1, v_2)$: Signature uniqueness score

$$U(v_1, v_2) = 1 - \frac{|\{u \in V_2 : C(v_1, u) > \theta\}| + |\{u \in V_1 : C(u, v_2) > \theta\}|}{|V_1| + |V_2|}$$

- $I(v_1, v_2)$: Structural importance (centrality-based)

$$I(v_1, v_2) = \sqrt{\frac{BC(v_1)}{max_{u \in V_1}BC(u)} \cdot \frac{BC(v_2)}{max_{u \in V_2}BC(u)}}$$

- $E(v_1, v_2)$: Expansion potential

$$E(v_1, v_2) = \frac{|N(v_1)| \cdot |N(v_2)|}{max_{u_1 \in V_1, u_2 \in V_2}|N(u_1)| \cdot |N(u_2)|}$$

- $K(v_1, v_2)$: Connectivity score based on local topology

where:

- $\theta \in [0, 1]$: Similarity threshold
- $BC(v)$: Betweenness centrality of node $v$
- $N(v)$: Set of immediate neighbors of node $v$

#### 4. Matching Score

Let $M \subseteq V_1 \times V_2$ be a partial matching between graphs. During bidirectional A\* search, matches are scored:

$$f(M) = g(M) + h(M)$$

The current match quality $g(M)$ is:

$$g(M) = \sum_{(v_1, v_2) \in M} C(v_1, v_2) \cdot Q(M, v_1, v_2)$$

where $Q(M, v_1, v_2)$ is the structural consistency term:

$$Q(M, v_1, v_2) = \frac{|\{(u_1, u_2) \in M : d(v_1, u_1) = d(v_2, u_2)\}|}{|M| - 1}$$

The heuristic estimate $h(M)$ is:

$$h(M) = \min(|V_1|, |V_2|) - |M| + \sum_{v_1 \in V_1 \setminus M_1} \max_{v_2 \in V_2 \setminus M_2} C(v_1, v_2)$$

where:

- $M_1 = \{v_1 \in V_1 : \exists v_2 \in V_2, (v_1, v_2) \in M\}$: Matched nodes from graph 1
- $M_2 = \{v_2 \in V_2 : \exists v_1 \in V_1, (v_1, v_2) \in M\}$: Matched nodes from graph 2
- $d(u, v)$: Shortest path distance between nodes $u$ and $v$ in their respective graphs

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
