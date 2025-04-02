"""Generator for benchmark datasets."""

import json
import os
import random
import uuid
import time
import signal
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdFMCS, rdMolDescriptors
import scipy.io

from prism.benchmark.configs.benchmark_config import BenchmarkConfig, CategoryConfig
from prism.core.molecular_pair import (
    BenchmarkDataset,
    KnownSolution,
    MolecularPair,
)
from prism.core.molecular_graph import MolecularGraph


class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""

    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out operations.

    Args:
        seconds: Number of seconds before timeout
    """

    def handler(signum, frame):
        raise TimeoutException("Operation timed out")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)

    # Set the alarm
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def validate_molecule(mol: Chem.Mol) -> bool:
    """Validate a molecule for common chemical issues.

    Args:
        mol: RDKit molecule to validate

    Returns:
        bool: True if molecule is valid, False otherwise
    """
    if mol is None:
        return False

    try:
        # Check for valid valence
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL)

        # Additional validation checks
        for atom in mol.GetAtoms():
            # Check for valid valence states
            if atom.GetExplicitValence() > atom.GetMaxValence():
                return False

            # Check for reasonable formal charges
            if abs(atom.GetFormalCharge()) > 1:
                return False

            # Validate common atom types
            if atom.GetSymbol() in ["O", "N", "F", "Cl", "Br", "I"]:
                if atom.GetExplicitValence() > atom.GetMaxValence():
                    return False

        return True

    except Exception:
        return False


class MoleculePool:
    """Pool of molecules classified by size and complexity."""

    def __init__(self):
        """Initialize an empty molecule pool."""
        self.molecules = {
            "tiny": [],
            "small": [],
            "medium": [],
            "large": [],
            "xlarge": [],
        }
        self.loaded = False

    def classify_molecule(self, mol: Chem.Mol, heavy_atom_count: int) -> str:
        """Classify a molecule by size based on heavy atom count.

        Args:
            mol: Molecule to classify
            heavy_atom_count: Number of heavy atoms in the molecule

        Returns:
            Size category ('tiny', 'small', 'medium', 'large', 'xlarge')
        """
        if heavy_atom_count <= 20:
            return "tiny"
        elif heavy_atom_count <= 50:
            return "small"
        elif heavy_atom_count <= 100:
            return "medium"
        elif heavy_atom_count <= 200:
            return "large"
        else:
            return "xlarge"

    def add_molecule(self, mol: Chem.Mol, heavy_atom_count: int) -> None:
        """
        Add a molecule to the pool.

        Args:
            mol: RDKit molecule to add
            heavy_atom_count: Number of heavy atoms in the molecule
        """
        if mol is None:
            return

        # Generate unique ID for molecule
        mol_id = str(uuid.uuid4())

        # Classify molecule based on heavy atom count
        size_category = self.classify_molecule(mol, heavy_atom_count)

        # Add to appropriate category
        if size_category not in self.molecules:
            self.molecules[size_category] = []
        self.molecules[size_category].append((mol_id, mol))

    def load_dataset(self, dataset_path: str) -> int:
        """
        Load molecules from a PubChem CSV file.

        Args:
            dataset_path: Path to the CSV file containing molecule data.

        Returns:
            Number of successfully loaded molecules.
        """
        loaded_count = 0

        with open(dataset_path, "r") as f:
            # Read and validate header
            header = next(f).strip().split(",")
            try:
                cid_idx = header.index("Compound CID")
                name_idx = header.index("Name")
                complexity_idx = header.index("Complexity")
                heavy_atom_idx = header.index("Heavy Atom Count")
                smiles_idx = header.index("SMILES")
            except ValueError as e:
                print(f"Error: Invalid CSV format - missing required column: {str(e)}")
                return loaded_count

            for line in f:
                try:
                    # Split on comma but preserve commas within quotes
                    fields = []
                    current_field = []
                    in_quotes = False
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == "," and not in_quotes:
                            fields.append("".join(current_field).strip())
                            current_field = []
                        else:
                            current_field.append(char)
                    fields.append("".join(current_field).strip())

                    if len(fields) < len(header):
                        continue

                    # Extract fields using validated indices
                    cid = fields[cid_idx]
                    name = fields[name_idx].strip('"')  # Remove any surrounding quotes
                    smiles = fields[smiles_idx].strip(
                        '"'
                    )  # Remove any surrounding quotes

                    # Clean up and validate heavy atom count
                    try:
                        heavy_atom_count = fields[heavy_atom_idx]
                        # Remove any text/units and convert to integer
                        heavy_atom_count = "".join(
                            c for c in heavy_atom_count if c.isdigit() or c == "."
                        )
                        heavy_atom_count = int(float(heavy_atom_count))
                    except (ValueError, IndexError):
                        print(
                            f"Error processing molecule: invalid heavy atom count - {fields[heavy_atom_idx]}"
                        )
                        continue

                    # Skip if SMILES is actually a number (likely a misaligned field)
                    if smiles.replace(".", "").isdigit():
                        print(f"Error processing molecule: invalid SMILES - {smiles}")
                        continue

                    # Create RDKit molecule from SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Error processing molecule: invalid SMILES - {smiles}")
                        continue

                    # Verify heavy atom count matches RDKit calculation
                    rdkit_heavy_count = mol.GetNumHeavyAtoms()
                    if rdkit_heavy_count != heavy_atom_count:
                        print(
                            f"Warning: Heavy atom count mismatch - CSV: {heavy_atom_count}, RDKit: {rdkit_heavy_count}"
                        )
                        heavy_atom_count = rdkit_heavy_count

                    # Add molecule to pool
                    size_category = self.classify_molecule(mol, heavy_atom_count)
                    if size_category not in self.molecules:
                        self.molecules[size_category] = []
                    self.molecules[size_category].append((smiles, mol))
                    loaded_count += 1

                    if loaded_count % 100 == 0:
                        print(f"Loaded {loaded_count} molecules")

                except Exception as e:
                    print(f"Error processing molecule: {str(e)}")
                    continue

            self.loaded = loaded_count > 0
            return loaded_count

    def get_random_molecule(self, size_category: str) -> Optional[Tuple[str, Chem.Mol]]:
        """Get a random molecule from the specified category."""
        if size_category not in self.molecules:
            return None

        category_molecules = self.molecules[size_category]
        if not category_molecules:
            return None

        return random.choice(category_molecules)


class DatasetGenerator(MoleculePool):
    """Generator for benchmark datasets."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the dataset generator.

        Args:
            config: Benchmark configuration
        """
        super().__init__()
        self.config = config
        self.dataset = BenchmarkDataset(
            name=config.name, description=config.description, pairs=[]
        )
        self.start_time = time.time()
        self.total_pairs = len(config.categories) * config.pairs_per_category
        self.generated_pairs = 0
        self.generation_timeout = 30

        # Initialize pair generators for each benchmark category
        self.pair_generators = {
            "SB-Small": lambda: self.generate_subgraph_benchmark_pair("small"),
            "SB-Medium": lambda: self.generate_subgraph_benchmark_pair("medium"),
            "SB-Large": lambda: self.generate_subgraph_benchmark_pair("large"),
            "SB-XLarge": lambda: self.generate_subgraph_benchmark_pair("xlarge"),
            "SI-Small/Medium": lambda: self.generate_size_impact_pair("small/medium"),
            "SI-Small/Large": lambda: self.generate_size_impact_pair("small/large"),
            "SI-Medium/Large": lambda: self.generate_size_impact_pair("medium/large"),
            "SI-Tiny/XLarge": lambda: self.generate_size_impact_pair("tiny/xlarge"),
            "TV-Linear": lambda: self.generate_topology_variation_pair("linear"),
            "TV-Branched": lambda: self.generate_topology_variation_pair("branched"),
            "TV-Cyclic": lambda: self.generate_topology_variation_pair("cyclic"),
            "TV-Mixed": lambda: self.generate_topology_variation_pair("mixed"),
            "SC-Low": lambda: self.generate_symmetry_challenge_pair("low"),
            "SC-Medium": lambda: self.generate_symmetry_challenge_pair("medium"),
            "SC-High": lambda: self.generate_symmetry_challenge_pair("high"),
            "SC-Mixed": lambda: self.generate_symmetry_challenge_pair("mixed"),
            "CS-Bottleneck": lambda: self.generate_corner_case_pair("bottleneck"),
            "CS-Symmetry": lambda: self.generate_corner_case_pair("symmetry"),
            "CS-NearMiss": lambda: self.generate_corner_case_pair("near_miss"),
            "CS-Deceptive": lambda: self.generate_corner_case_pair("deceptive"),
        }

    def load_molecular_dataset(self, dataset_path: str) -> int:
        """Load molecules from a dataset file.

        Args:
            dataset_path: Path to the dataset file (.smi or .csv format)

        Returns:
            Number of molecules loaded
        """
        return super().load_dataset(dataset_path)

    def load_benchmark_dataset(self, file_path: str) -> BenchmarkDataset:
        """Load a benchmark dataset from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded benchmark dataset
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Create dataset
        dataset = BenchmarkDataset(
            name=data["name"], description=data["description"], pairs=[]
        )

        # Load pairs
        for pair_dict in data["pairs"]:
            # Handle both singular and plural optimal solution formats
            optimal_solution = None
            if "optimal_solution" in pair_dict:
                # Single solution format
                optimal_solution = KnownSolution(**pair_dict["optimal_solution"])
            elif "optimal_solutions" in pair_dict:
                # Multiple solutions format (legacy)
                optimal_solutions = [
                    KnownSolution(**sol) for sol in pair_dict["optimal_solutions"]
                ]
                if optimal_solutions:
                    optimal_solution = optimal_solutions[0]

            # If no solution was found, create a default one
            if optimal_solution is None:
                optimal_solution = KnownSolution()

            # Create pair
            pair = MolecularPair(
                id=pair_dict["id"],
                category=pair_dict["category"],
                smiles1=pair_dict["smiles1"],
                smiles2=pair_dict["smiles2"],
                mol1_size=pair_dict["mol1_size"],
                mol2_size=pair_dict["mol2_size"],
                max_common_size=pair_dict["max_common_size"],
                overlap_ratio=pair_dict["overlap_ratio"],
                false_leads_count=pair_dict["false_leads_count"],
                optimal_solution=optimal_solution,
                structural_features=pair_dict.get("structural_features", {}),
            )

            dataset.pairs.append(pair)

        return dataset

    def generate_dataset(self) -> BenchmarkDataset:
        """Generate the full benchmark dataset according to the configuration.

        Returns:
            Complete benchmark dataset
        """
        self.start_time = time.time()
        print(f"Starting dataset generation for {self.total_pairs} pairs...")

        for category_id, category_config in self.config.categories.items():
            print(f"\nGenerating pairs for category: {category_id}")
            # Generate pairs for this category
            for i in range(self.config.pairs_per_category):
                pair = self._generate_pair(category_id, category_config)
                if pair:
                    self.dataset.pairs.append(pair)
                    self.generated_pairs += 1
                    self._report_progress()

        elapsed_time = time.time() - self.start_time
        print(f"\nDataset generation completed in {elapsed_time:.2f} seconds")
        print(
            f"Generated {len(self.dataset.pairs)} pairs out of {self.total_pairs} requested"
        )
        return self.dataset

    def _report_progress(self):
        """Report progress of dataset generation."""
        elapsed_time = time.time() - self.start_time
        progress = (self.generated_pairs / self.total_pairs) * 100
        pairs_per_second = (
            self.generated_pairs / elapsed_time if elapsed_time > 0 else 0
        )

        print(
            f"\rProgress: {progress:.1f}% ({self.generated_pairs}/{self.total_pairs}) "
            f"[{pairs_per_second:.1f} pairs/s]",
            end="",
            flush=True,
        )

    def _generate_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a single molecular pair for a specific category.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        # Implementation depends on category type
        if category_id.startswith("SB-"):
            # Size-balanced category
            return self._generate_size_balanced_pair(category_id, category_config)
        elif category_id.startswith("SI-"):
            # Size-imbalanced category
            return self._generate_size_imbalanced_pair(category_id, category_config)
        elif category_id.startswith("TV-"):
            # Topological variation category
            return self._generate_topological_pair(category_id, category_config)
        elif category_id.startswith("SC-"):
            # Signature complexity category
            return self._generate_signature_complexity_pair(
                category_id, category_config
            )
        elif category_id.startswith("CS-"):
            # Challenge set category
            return self._generate_challenge_pair(category_id, category_config)
        else:
            # Unknown category
            print(f"Unknown category type: {category_id}")
            return None

    def _generate_size_balanced_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a size-balanced molecular pair.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        if not self.loaded:
            print("No molecular dataset loaded. Using fallback pair.")
            return self._generate_fallback_pair(category_id)

        # Determine size categories based on configuration
        size_category = category_id.split("-")[1].lower()  # e.g., "SB-Small" -> "small"

        # Get random molecules of appropriate sizes
        mol_data = self.get_random_molecule(size_category)
        if not mol_data:
            print(f"No molecules available in category {size_category}")
            return self._generate_fallback_pair(category_id)

        smiles1, mol1 = mol_data

        # Try to find a second molecule with similar size
        for _ in range(5):  # Try 5 times
            mol_data2 = self.get_random_molecule(size_category)
            if not mol_data2:
                continue

            smiles2, mol2 = mol_data2
            if smiles1 == smiles2:  # Don't use the same molecule
                continue

            # Find the MCS to determine the mapping
            mcs = rdFMCS.FindMCS(
                [mol1, mol2],
                completeRingsOnly=True,
                ringMatchesRingOnly=True,
                matchValences=True,
                timeout=5,  # 5 second timeout
            )

            if mcs.numAtoms < 3:  # If MCS is too small
                continue

            # Create mapping between the molecules
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            match1 = mol1.GetSubstructMatch(mcs_mol)
            match2 = mol2.GetSubstructMatch(mcs_mol)

            if not match1 or not match2:
                continue

            # Create the mapping
            mapping = {}
            for i, idx1 in enumerate(match1):
                idx2 = match2[i]
                mapping[idx1] = idx2

            # Calculate actual properties
            mol1_size = mol1.GetNumAtoms()
            mol2_size = mol2.GetNumAtoms()
            overlap_size = len(mapping)
            overlap_ratio = overlap_size / min(mol1_size, mol2_size)

            # Check if the overlap ratio is within desired range
            if not (
                category_config.overlap.min_overlap_ratio
                <= overlap_ratio
                <= category_config.overlap.max_overlap_ratio
            ):
                continue

            # Generate some false leads
            false_leads_count = random.randint(
                category_config.min_false_leads, category_config.max_false_leads
            )

            # Create the optimal solution
            optimal_solution = KnownSolution(
                mapping=mapping, size=len(mapping), score=1.0, is_optimal=True
            )

            # Create unique ID for this pair
            pair_id = f"{category_id}-{uuid.uuid4().hex[:8]}"

            # Create and return the molecular pair
            return MolecularPair(
                id=pair_id,
                category=category_id,
                smiles1=smiles1,
                smiles2=smiles2,
                mol1=mol1,
                mol2=mol2,
                mol1_size=mol1_size,
                mol2_size=mol2_size,
                max_common_size=overlap_size,
                overlap_ratio=overlap_ratio,
                false_leads_count=false_leads_count,
                optimal_solution=optimal_solution,
                structural_features=self._calculate_structural_features(mol1, mol2),
            )

        # If we get here, we failed all attempts
        return self._generate_fallback_pair(category_id)

    def _generate_fallback_pair(self, category_id: str) -> MolecularPair:
        """Generate a fallback molecular pair when regular generation fails.

        Args:
            category_id: ID of the category

        Returns:
            A simple predefined MolecularPair
        """
        # Use benzene and toluene as a simple example
        smiles1 = "c1ccccc1"  # Benzene
        smiles2 = "c1ccccc1C"  # Toluene

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # Find the MCS mapping
        mcs = rdFMCS.FindMCS(
            [mol1, mol2],
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            matchValences=True,
        )

        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        match1 = mol1.GetSubstructMatch(mcs_mol)
        match2 = mol2.GetSubstructMatch(mcs_mol)

        # Create the mapping
        mapping = {}
        for i, idx1 in enumerate(match1):
            idx2 = match2[i]
            mapping[idx1] = idx2

        # Create the optimal solution
        optimal_solution = KnownSolution(
            mapping=mapping, size=len(mapping), score=1.0, is_optimal=True
        )

        # Create unique ID
        pair_id = f"{category_id}-fallback-{uuid.uuid4().hex[:6]}"

        # Create the molecular pair
        return MolecularPair(
            id=pair_id,
            category=category_id,
            smiles1=smiles1,
            smiles2=smiles2,
            mol1=mol1,
            mol2=mol2,
            mol1_size=mol1.GetNumAtoms(),
            mol2_size=mol2.GetNumAtoms(),
            max_common_size=len(mapping),
            overlap_ratio=len(mapping) / mol1.GetNumAtoms(),
            false_leads_count=0,
            optimal_solution=optimal_solution,
            structural_features=self._calculate_structural_features(mol1, mol2),
        )

    def _generate_size_imbalanced_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a size-imbalanced molecular pair.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        if not self.loaded:
            return self._generate_fallback_pair(category_id)

        # Parse size categories from category ID (e.g., "SI-Small/Medium")
        size_cats = category_id.split("-")[1].lower().split("/")
        if len(size_cats) != 2:
            return self._generate_fallback_pair(category_id)

        # Get molecules of different sizes
        mol_data1 = self.get_random_molecule(size_cats[0])
        mol_data2 = self.get_random_molecule(size_cats[1])

        if not mol_data1 or not mol_data2:
            return self._generate_fallback_pair(category_id)

        smiles1, mol1 = mol_data1
        smiles2, mol2 = mol_data2

        # Find the MCS
        mcs = rdFMCS.FindMCS(
            [mol1, mol2],
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            matchValences=True,
            timeout=5,
        )

        if mcs.numAtoms < 3:
            return self._generate_fallback_pair(category_id)

        # Create mapping
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        match1 = mol1.GetSubstructMatch(mcs_mol)
        match2 = mol2.GetSubstructMatch(mcs_mol)

        if not match1 or not match2:
            return self._generate_fallback_pair(category_id)

        # Create the mapping
        mapping = {}
        for i, idx1 in enumerate(match1):
            idx2 = match2[i]
            mapping[idx1] = idx2

        # Calculate properties
        mol1_size = mol1.GetNumAtoms()
        mol2_size = mol2.GetNumAtoms()
        overlap_size = len(mapping)
        overlap_ratio = overlap_size / min(mol1_size, mol2_size)

        # Generate false leads
        false_leads_count = random.randint(
            category_config.min_false_leads, category_config.max_false_leads
        )

        # Create optimal solution
        optimal_solution = KnownSolution(
            mapping=mapping, size=len(mapping), score=1.0, is_optimal=True
        )

        # Create unique ID
        pair_id = f"{category_id}-{uuid.uuid4().hex[:8]}"

        # Create and return the pair
        return MolecularPair(
            id=pair_id,
            category=category_id,
            smiles1=smiles1,
            smiles2=smiles2,
            mol1=mol1,
            mol2=mol2,
            mol1_size=mol1_size,
            mol2_size=mol2_size,
            max_common_size=overlap_size,
            overlap_ratio=overlap_ratio,
            false_leads_count=false_leads_count,
            optimal_solution=optimal_solution,
            structural_features=self._calculate_structural_features(mol1, mol2),
        )

    def _generate_topological_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a pair with specific topological characteristics.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        # For demonstration, we'll just use the balanced pair generation
        # In a real implementation, this would use more sophisticated methods
        # to control the topology of the molecules
        return self._generate_size_balanced_pair(category_id, category_config)

    def _generate_signature_complexity_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a pair with controlled signature complexity.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        # For demonstration, we'll just use the balanced pair generation
        # In a real implementation, this would control atom types and connectivity
        return self._generate_size_balanced_pair(category_id, category_config)

    def _generate_challenge_pair(
        self, category_id: str, category_config: CategoryConfig
    ) -> Optional[MolecularPair]:
        """Generate a challenging test case.

        Args:
            category_id: ID of the category
            category_config: Configuration for the category

        Returns:
            MolecularPair instance or None if generation failed
        """
        # For demonstration, we'll just use the balanced pair generation
        # In a real implementation, this would generate specific challenges
        return self._generate_size_balanced_pair(category_id, category_config)

    def _generate_molecular_fragment(self, target_size: int) -> Optional[str]:
        """Generate a molecular fragment of approximate target size.

        Args:
            target_size: Target number of atoms

        Returns:
            SMILES string of generated fragment or None if generation failed
        """
        # Define building blocks for different sizes with diverse atom types
        blocks = {
            "small": [
                "CC",  # Ethane
                "CCO",  # Ethanol
                "CCN",  # Ethylamine
                "CO",  # Methanol
                "CN",  # Methylamine
                "CC(=O)O",  # Acetic acid
                "CC(=O)N",  # Acetamide
                "NC=O",  # Formamide
                "C=O",  # Formaldehyde
                "CNO",  # N-methylhydroxylamine
            ],
            "medium": [
                "C1CCCCC1",  # Cyclohexane
                "C1CCNCC1",  # Piperidine
                "C1COCCC1",  # Tetrahydropyran
                "CC(=O)CCO",  # 4-hydroxy-2-butanone
                "CC(=O)CN",  # Aminoacetone
                "CC(=O)OC",  # Methyl acetate
                "CC(=O)NC",  # N-methylacetamide
                "OCC(O)CO",  # Glycerol
                "NCCO",  # Ethanolamine
            ],
            "large": [
                "C1CCC2CCCCC2C1",  # Decalin
                "C1CC2CCC1CC2",  # Bicyclo[2.2.2]octane
                "C1CCCCC1NC=O",  # N-cyclohexylformamide
                "CC(=O)C1CCCCC1",  # Cyclohexyl methyl ketone
                "OCC1CCCCC1",  # Cyclohexylmethanol
                "CC(=O)CCCCO",  # 5-hydroxypentanone
                "OCCCCCO",  # 1,5-pentanediol
                "NCCCCCCN",  # 1,6-hexanediamine
            ],
            "xlarge": [
                "C1CC2CCC3CCC4CCC1CC2CC34",  # Twistane
                "OCCCCCCCCCCCO",  # 1,10-decanediol
                "CC(=O)CCCCCCCCCO",  # 10-hydroxydecanone
                "NCCCCCCCCCCN",  # 1,10-decanediamine
                "OCC1CCC2CCCCC2C1",  # Decalinylmethanol
            ],
        }

        # Select appropriate block size based on target size
        if target_size <= 10:
            block_size = "small"
        elif target_size <= 20:
            block_size = "medium"
        elif target_size <= 40:
            block_size = "large"
        else:
            block_size = "xlarge"

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                with timeout(self.generation_timeout):
                    # Select a random block
                    base_block = random.choice(blocks[block_size])
                    mol = Chem.MolFromSmiles(base_block)

                    if not validate_molecule(mol):
                        continue

                    # If we need a larger molecule, try to extend it
                    current_size = mol.GetNumAtoms()
                    if current_size < target_size:
                        # Try to extend with smaller blocks that include heteroatoms
                        extension_blocks = blocks["small"]
                        for _ in range(min(3, target_size - current_size)):
                            ext_block = random.choice(extension_blocks)
                            ext_mol = Chem.MolFromSmiles(ext_block)

                            if not validate_molecule(ext_mol):
                                continue

                            # Try to combine molecules safely
                            try:
                                combined = Chem.CombineMols(mol, ext_mol)
                                if not validate_molecule(combined):
                                    continue
                                mol = combined
                            except:
                                continue

                    # Convert back to SMILES and validate
                    try:
                        result = Chem.MolToSmiles(mol)
                        if result and validate_molecule(Chem.MolFromSmiles(result)):
                            return result
                    except:
                        continue

            except TimeoutException:
                print(f"Fragment generation attempt {attempt + 1} timed out")
                continue
            except Exception as e:
                print(f"Error generating fragment: {e}")
                continue

        return None

    def _extend_to_target_size(
        self, base_smiles: str, target_size: int
    ) -> Optional[str]:
        """Extend a molecular fragment to reach a target size.

        Args:
            base_smiles: SMILES of the base fragment
            target_size: Target number of atoms

        Returns:
            SMILES string of extended molecule or None if extension failed
        """
        MAX_ATTEMPTS = 3
        MAX_DEPTH = 3

        # Simple extension blocks that maintain valid valence
        blocks = {
            "small": [
                "C",  # Methyl
                "CC",  # Ethyl
            ],
            "medium": [
                "C1CCCC1",  # Cyclopentane
                "CCCC",  # Butane
            ],
        }

        def try_extend(smiles: str, target: int, depth: int = 0) -> Optional[str]:
            if depth >= MAX_DEPTH:
                return None

            try:
                with timeout(self.generation_timeout):
                    mol = Chem.MolFromSmiles(smiles)
                    if not validate_molecule(mol):
                        return None

                    current_size = mol.GetNumAtoms()
                    if current_size >= target:
                        return smiles

                    space_left = target - current_size

                    # Try blocks in order of size preference
                    for size in ["medium", "small"]:
                        if size not in blocks:
                            continue

                        for block in blocks[size]:
                            block_mol = Chem.MolFromSmiles(block)
                            if not block_mol or block_mol.GetNumAtoms() > space_left:
                                continue

                            try:
                                # Try adding the block
                                result = smiles + block
                                result_mol = Chem.MolFromSmiles(result)
                                if not validate_molecule(result_mol):
                                    continue

                                result_size = result_mol.GetNumAtoms()

                                # If we're close enough to target size, return
                                if abs(result_size - target) <= 5:
                                    return result

                                # If we need more atoms, try extending further
                                if result_size < target:
                                    next_result = try_extend(result, target, depth + 1)
                                    if next_result:
                                        return next_result
                            except:
                                continue

                    # If no blocks worked, try adding a simple chain
                    try:
                        chain_length = min(
                            space_left, 2
                        )  # More conservative chain length
                        result = smiles + "C" * chain_length
                        result_mol = Chem.MolFromSmiles(result)
                        if validate_molecule(result_mol):
                            return result
                    except:
                        pass

                    return None

            except TimeoutException:
                print(f"Extension operation timed out at depth {depth}")
                return None
            except Exception as e:
                print(f"Error in extension: {e}")
                return None

        # Try extension with multiple attempts
        for attempt in range(MAX_ATTEMPTS):
            try:
                with timeout(self.generation_timeout):
                    result = try_extend(base_smiles, target_size)
                    if result and validate_molecule(Chem.MolFromSmiles(result)):
                        return result
            except TimeoutException:
                print(f"Extension attempt {attempt + 1} timed out")
                continue
            except Exception as e:
                print(f"Extension attempt {attempt + 1} failed: {str(e)}")
                continue

        # If all attempts fail, try a simple chain extension
        try:
            with timeout(self.generation_timeout):
                result = base_smiles + "C" * min(2, target_size - len(base_smiles))
                if validate_molecule(Chem.MolFromSmiles(result)):
                    return result
        except (TimeoutException, Exception):
            pass

        return None

    def _generate_suboptimal_solution(
        self,
        mol1: Chem.Mol,
        mol2: Chem.Mol,
        optimal: KnownSolution,
        mol1_size: int,
        mol2_size: int,
    ) -> Optional[KnownSolution]:
        """Generate a suboptimal solution by perturbing the optimal one.

        Args:
            mol1: First molecule
            mol2: Second molecule
            optimal: Optimal solution
            mol1_size: Size of first molecule
            mol2_size: Size of second molecule

        Returns:
            Suboptimal solution or None if generation failed
        """
        # Create a perturbed mapping from the optimal one
        # This is a simple approach - real implementation would be more sophisticated

        # Start with the optimal mapping
        mapping = optimal.mapping.copy()

        # Determine how many mappings to change
        change_ratio = random.uniform(0.1, 0.4)  # Change 10-40% of the mapping
        num_changes = max(1, int(len(mapping) * change_ratio))

        # Remove some mappings
        keys_to_remove = random.sample(list(mapping.keys()), num_changes)
        for key in keys_to_remove:
            del mapping[key]

        # Add some incorrect mappings if any nodes are unmapped
        unmapped_mol1 = set(range(mol1_size)) - set(mapping.keys())
        unmapped_mol2 = set(range(mol2_size)) - set(mapping.values())

        # Add a few random incorrect mappings
        num_additions = min(len(unmapped_mol1), len(unmapped_mol2), num_changes)
        mol1_to_add = random.sample(list(unmapped_mol1), num_additions)
        mol2_to_add = random.sample(list(unmapped_mol2), num_additions)

        for i in range(num_additions):
            mapping[mol1_to_add[i]] = mol2_to_add[i]

        # Calculate score (size compared to optimal)
        score = len(mapping) / len(optimal.mapping) * 0.8  # Cap at 80% of optimal score

        return KnownSolution(
            mapping=mapping, size=len(mapping), score=score, is_optimal=False
        )

    def _calculate_structural_features(
        self, mol1: Chem.Mol, mol2: Chem.Mol
    ) -> Dict[str, Any]:
        """Calculate structural features for the molecular pair.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Dictionary of structural features
        """
        # In a real implementation, would calculate various graph-theoretic properties
        # For demonstration, we'll return some basic RDKit descriptors

        features = {
            "mol1": {
                "num_atoms": mol1.GetNumAtoms(),
                "num_bonds": mol1.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol1),
                "fraction_sp3": rdMolDescriptors.CalcFractionCSP3(mol1),
                "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol1),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol1),
            },
            "mol2": {
                "num_atoms": mol2.GetNumAtoms(),
                "num_bonds": mol2.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol2),
                "fraction_sp3": rdMolDescriptors.CalcFractionCSP3(mol2),
                "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol2),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol2),
            },
        }

        return features

    def save_dataset(self, file_path: str) -> None:
        """Save the dataset to a JSON file.

        Args:
            file_path: Path to save the dataset
        """
        # Convert to dictionary format
        dataset_dict = {
            "name": self.dataset.name,
            "description": self.dataset.description,
            "pairs": [],
        }

        # Convert each pair
        for pair in self.dataset.pairs:
            # Convert to SMILES to ensure serialization
            pair_dict = pair.dict(exclude={"mol1", "mol2"})
            dataset_dict["pairs"].append(pair_dict)

        # Save to file
        with open(file_path, "w") as f:
            json.dump(dataset_dict, f, indent=2)

    def _create_molecular_pair(
        self,
        category_id: str,
        smiles1: str,
        smiles2: str,
        optimal_solution: KnownSolution,
        false_leads_count: int = 0,
        structural_features: Optional[Dict[str, Any]] = None,
    ) -> MolecularPair:
        """Create a molecular pair for benchmark dataset.

        Args:
            category_id: Category identifier
            smiles1: SMILES string for molecule 1
            smiles2: SMILES string for molecule 2
            optimal_solution: Known optimal solution
            false_leads_count: Number of significant suboptimal solutions
            structural_features: Optional structural features

        Returns:
            MolecularPair object
        """
        # Create unique ID for the pair
        pair_id = f"{category_id}-{self._generate_random_id()}"

        # Calculate molecule sizes
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        mol1_size = mol1.GetNumAtoms()
        mol2_size = mol2.GetNumAtoms()

        # Create the molecular pair
        return MolecularPair(
            id=pair_id,
            category=category_id,
            smiles1=smiles1,
            smiles2=smiles2,
            mol1_size=mol1_size,
            mol2_size=mol2_size,
            max_common_size=optimal_solution.size,
            overlap_ratio=optimal_solution.size / min(mol1_size, mol2_size),
            false_leads_count=false_leads_count,
            optimal_solution=optimal_solution,
            structural_features=structural_features or {},
        )

    def _generate_random_id(self, length: int = 8) -> str:
        """Generate a random ID string.

        Args:
            length: Length of the ID string

        Returns:
            Random ID string
        """
        import random
        import string

        # Generate random hex string
        return "".join(random.choices(string.hexdigits.lower(), k=length))

    def generate_subgraph_benchmark_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for subgraph benchmark.

        Args:
            category: Category to generate pair for (e.g. 'SB-Small')

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) or None if generation failed
        """
        size = category.split("-")[1].lower()
        mol = self.get_random_molecule(size)
        if mol is None:
            print(f"No molecules available in category {size}")
            return None
        return mol, mol

    def generate_size_impact_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for size impact benchmark.

        Args:
            category: Combined category string (e.g. 'small/medium')

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) or None if generation failed
        """
        sizes = category.split("/")
        if len(sizes) != 2:
            print(f"Invalid size impact category format: {category}")
            return None

        mol1 = self.get_random_molecule(sizes[0])
        mol2 = self.get_random_molecule(sizes[1])
        if mol1 is None or mol2 is None:
            print(f"Could not get molecules for sizes {sizes[0]} and {sizes[1]}")
            return None
        return mol1, mol2

    def generate_topology_variation_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for topology variation benchmark.

        Args:
            category: Category to generate pair for (e.g. 'TV-Linear')

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) or None if generation failed
        """
        # For now, just return two random small molecules
        mol1 = self.get_random_molecule("small")
        mol2 = self.get_random_molecule("small")
        if mol1 is None or mol2 is None:
            print("Could not get molecules for topology variation")
            return None
        return mol1, mol2

    def generate_symmetry_challenge_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for symmetry challenge benchmark.

        Args:
            category: Category to generate pair for (e.g. 'SC-Low')

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) or None if generation failed
        """
        # For now, just return two random small molecules
        mol1 = self.get_random_molecule("small")
        mol2 = self.get_random_molecule("small")
        if mol1 is None or mol2 is None:
            print("Could not get molecules for symmetry challenge")
            return None
        return mol1, mol2

    def generate_corner_case_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for corner case benchmark.

        Args:
            category: Category to generate pair for (e.g. 'CS-Symmetry')

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) or None if generation failed
        """
        # For now, just return two random small molecules
        mol1 = self.get_random_molecule("small")
        mol2 = self.get_random_molecule("small")
        if mol1 is None or mol2 is None:
            print("Could not get molecules for corner case")
            return None
        return mol1, mol2

    def generate_pair(
        self, category: str
    ) -> Optional[Tuple[Tuple[str, Chem.Mol], Tuple[str, Chem.Mol]]]:
        """Generate a pair of molecules for the given category.

        Args:
            category: Category to generate pair for

        Returns:
            Tuple of ((id1, mol1), (id2, mol2)) if successful, None otherwise
        """
        if category not in self.pair_generators:
            return None

        generator = self.pair_generators[category]
        return generator()
