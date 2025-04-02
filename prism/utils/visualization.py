"""Visualization utilities for molecular graph matching results."""

from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

from prism.algorithm.graph_matcher import MatchResult


def visualize_match(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    match_result: MatchResult,
    highlight_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    output_file: Optional[str] = None,
    image_size: Tuple[int, int] = (600, 300),
    atom_labels: bool = True,
) -> Draw.Image:
    """Create a visualization of matching atoms between two molecules.

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        match_result: Matching result containing the mapping.
        highlight_colors: Optional dictionary mapping element symbols to RGB colors.
        output_file: Optional path to save the output image.
        image_size: Size of the output image (width, height).
        atom_labels: Whether to add atom labels.

    Returns:
        RDKit Image object containing the visualization.
    """
    # Default colors if not provided
    if highlight_colors is None:
        highlight_colors = {
            "C": (0.7, 0.7, 0.7),
            "N": (0.2, 0.2, 1.0),
            "O": (1.0, 0.2, 0.2),
            "S": (0.9, 0.9, 0.0),
            "F": (0.5, 1.0, 0.5),
            "Cl": (0.0, 0.8, 0.0),
            "Br": (0.6, 0.0, 0.0),
            "I": (0.5, 0.0, 0.5),
            "P": (1.0, 0.5, 0.0),
            "default": (0.7, 0.7, 0.7),
        }

    # Create copies of molecules to modify
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)

    # Get atom indices for matching atoms
    match_atoms1 = list(match_result.mapping.keys())
    match_atoms2 = list(match_result.mapping.values())

    # Create atom colors for highlighting
    atom_colors1 = {}
    atom_colors2 = {}

    # Create atom maps for visualization
    for i, (idx1, idx2) in enumerate(match_result.mapping.items()):
        # Get element symbols
        atom1 = mol1.GetAtomWithIdx(idx1)
        atom2 = mol2.GetAtomWithIdx(idx2)

        element1 = atom1.GetSymbol()
        element2 = atom2.GetSymbol()

        # Get colors for elements
        color1 = highlight_colors.get(element1, highlight_colors["default"])
        color2 = highlight_colors.get(element2, highlight_colors["default"])

        # Set colors
        atom_colors1[idx1] = color1
        atom_colors2[idx2] = color2

        # Add atom number labels if requested
        if atom_labels:
            mol1_copy.GetAtomWithIdx(idx1).SetProp("atomNote", str(i + 1))
            mol2_copy.GetAtomWithIdx(idx2).SetProp("atomNote", str(i + 1))

    # Calculate image size for each molecule
    mol_size = (image_size[0] // 2, image_size[1])

    # Draw molecules
    drawer = Draw.MolDrawOptions()
    drawer.useBWAtomPalette()

    img1 = Draw.MolToImage(
        mol1_copy,
        size=mol_size,
        highlightAtoms=match_atoms1,
        highlightAtomColors=atom_colors1,
        options=drawer,
    )
    img2 = Draw.MolToImage(
        mol2_copy,
        size=mol_size,
        highlightAtoms=match_atoms2,
        highlightAtomColors=atom_colors2,
        options=drawer,
    )

    # Create combined image
    combined_img = Draw.MolsToGridImage(
        [mol1_copy, mol2_copy],
        molsPerRow=2,
        subImgSize=mol_size,
        highlightAtomLists=[match_atoms1, match_atoms2],
        highlightAtomColors=[atom_colors1, atom_colors2],
        legends=["Molecule 1", "Molecule 2"],
    )

    # Save image if requested
    if output_file:
        combined_img.save(output_file)

    return combined_img


def create_match_report(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    match_result: MatchResult,
    output_file: Optional[str] = None,
) -> str:
    """Create a detailed report of the matching result.

    Args:
        mol1: First molecule.
        mol2: Second molecule.
        match_result: Matching result.
        output_file: Optional path to save the report.

    Returns:
        Report text.
    """
    # Start report
    report = []
    report.append("PRISM Molecular Matching Result")
    report.append("=" * 30 + "\n")

    # Basic information
    mol1_formula = Chem.CalcMolFormula(mol1)
    mol2_formula = Chem.CalcMolFormula(mol2)

    mol1_smiles = Chem.MolToSmiles(mol1)
    mol2_smiles = Chem.MolToSmiles(mol2)

    report.append(f"Molecule 1: {mol1_formula} ({mol1_smiles})")
    report.append(f"Molecule 2: {mol2_formula} ({mol2_smiles})\n")

    # Match statistics
    report.append(f"Match size: {match_result.size} atoms")
    report.append(f"Match score: {match_result.score:.4f}")
    report.append(f"Processing time: {match_result.match_time:.4f} seconds\n")

    # Match coverage
    coverage1 = match_result.size / mol1.GetNumAtoms() * 100
    coverage2 = match_result.size / mol2.GetNumAtoms() * 100

    report.append(f"Coverage of molecule 1: {coverage1:.1f}%")
    report.append(f"Coverage of molecule 2: {coverage2:.1f}%\n")

    # Atom mapping details
    report.append("Atom mapping:")
    report.append("-" * 30)

    for atom1_idx, atom2_idx in match_result.mapping.items():
        atom1 = mol1.GetAtomWithIdx(atom1_idx)
        atom2 = mol2.GetAtomWithIdx(atom2_idx)

        atom1_info = f"{atom1.GetSymbol()}{atom1_idx+1}"
        atom2_info = f"{atom2.GetSymbol()}{atom2_idx+1}"

        report.append(f"  {atom1_info:10s} → {atom2_info:10s}")

    # Join report lines
    report_text = "\n".join(report)

    # Save report if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)

    return report_text
