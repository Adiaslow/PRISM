# analyze_atoms.py
"""Script to analyze atom distribution in generated benchmark dataset."""

from collections import Counter

from rdkit import Chem

from prism.benchmark.configs.benchmark_config import BenchmarkConfig
from prism.benchmark.services.dataset_generator import DatasetGenerator


def main():
    """Analyze atom distribution in generated dataset."""
    # Create configuration and generate dataset
    print("Generating dataset...")
    config = BenchmarkConfig.default_config()
    gen = DatasetGenerator(config)
    dataset = gen.generate_dataset()

    # Collect all atoms from all molecules
    all_atoms = []
    for pair in dataset.pairs:
        # Process first molecule
        mol1 = Chem.MolFromSmiles(pair.smiles1)
        if mol1:
            all_atoms.extend([atom.GetSymbol() for atom in mol1.GetAtoms()])

        # Process second molecule
        mol2 = Chem.MolFromSmiles(pair.smiles2)
        if mol2:
            all_atoms.extend([atom.GetSymbol() for atom in mol2.GetAtoms()])

    # Count and print distribution
    atom_counts = Counter(all_atoms)
    total_atoms = sum(atom_counts.values())

    print(f"\nAtom distribution in the dataset ({total_atoms} total atoms):")
    print("-" * 50)
    print(f"{'Atom':<6} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 50)

    for atom, count in atom_counts.most_common():
        percentage = count / total_atoms * 100
        print(f"{atom:<6} | {count:<8} | {percentage:.2f}%")


if __name__ == "__main__":
    main()
