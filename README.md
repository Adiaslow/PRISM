# PRISM: Parallel Recursive Isomorphism Search for Molecules

PRISM is a state-of-the-art algorithm for molecular graph matching, specifically designed to find maximum common substructures between molecules. It employs a sophisticated combination of techniques to efficiently identify structural similarities in chemical compounds.

## Key Features

### 1. Advanced Node Signature Generation

- Generates unique molecular fingerprints using:
  - Element type information
  - Local connectivity patterns
  - Neighborhood topology up to configurable distances
  - Cycle participation information
- Configurable weights for different signature components
- Support for bond types and cycle information

### 2. Intelligent Seed Selection

- Multi-factor seed pair selection based on:
  - Signature uniqueness (40%)
  - Structural importance (30%)
  - Expansion potential (20%)
  - Connectivity patterns (10%)
- Ensures diverse starting points for the search
- Optimizes for both local and global matching quality

### 3. Bidirectional A\* Search

- Uses a bidirectional search strategy with A\* guidance
- Employs multiple scoring factors:
  - Current contribution (30%)
  - Structural consistency (30%)
  - Expansion potential (20%)
  - Global impact (20%)
- Forward checking and conflict backjumping for efficiency
- Parallel processing support for improved performance

## Algorithm Parameters

### Signature Generation

```python
signature_params = {
    "max_distance": 2,
    "use_bond_types": True,
    "use_cycles": True
}
```

### Compatibility Matrix

```python
compatibility_params = {
    "element_match_required": True,
    "min_signature_similarity": 0.7,
    "bond_type_match_required": True,
    "progressive_refinement": True,
    "max_compatible_degree_diff": 1
}
```

### Matching Parameters

```python
match_params = {
    "max_time_seconds": 30,
    "max_iterations": 1000000,
    "num_threads": 4,
    "use_forward_checking": True,
    "use_conflict_backjumping": True
}
```

## Usage

### Basic Usage

```python
from prism import MolecularGraphMatcher
from rdkit import Chem

# Initialize the matcher
matcher = MolecularGraphMatcher(
    signature_params=signature_params,
    compatibility_params=compatibility_params,
    match_params=match_params
)

# Find maximum common substructure
mol1 = Chem.MolFromSmiles("CC(=O)O")
mol2 = Chem.MolFromSmiles("CC(=O)OH")
result = matcher.find_maximum_common_subgraph(mol1, mol2)

# Access results
mapping = result.mapping  # Node mapping between molecules
size = result.size       # Size of the common substructure
score = result.score     # Match quality score
time = result.match_time # Execution time
```

### Benchmark Usage

```python
from prism.benchmark import ComparativeBenchmarkRunner
from prism.benchmark.configs.benchmark_config import BenchmarkConfig

# Create benchmark configuration
config = BenchmarkConfig(
    name="Comparative Benchmark",
    description="Comparison of molecular subgraph isomorphism algorithms",
    pairs_per_category=20
)

# Run benchmark
runner = ComparativeBenchmarkRunner(
    datasets=dataset,
    num_workers=12,
    use_processes=True
)

result = runner.run_comparative_benchmark()
```

## Performance Characteristics

PRISM is optimized for:

- Medium to large organic molecules (10-100 atoms)
- Molecules with diverse structural features
- Parallel processing on multi-core systems
- Memory-efficient operation with large datasets

The algorithm provides a good balance between:

- Search completeness
- Execution speed
- Solution quality
- Memory usage

## Implementation Details

The algorithm is implemented in Python with key optimizations:

- Numpy-based signature computation
- NetworkX-based graph operations
- Parallel processing support
- Memory-efficient data structures
- RDKit integration for molecular operations

## License

[Insert your license information here]

## Citation

[Insert citation information here]
