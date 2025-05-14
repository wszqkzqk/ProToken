#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from Bio.PDB import PDBParser

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate the distance between specified atoms in two residues across MODEL frames in a PDB file, and save the plot."
    )
    parser.add_argument("pdb_file", help="Path to the input PDB file containing multiple MODEL frames")
    parser.add_argument("-c", "--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("--arg-res", type=int, default=131, help="Residue number for Arg (default: 131)")
    parser.add_argument("--asp-res", type=int, default=146, help="Residue number for Asp (default: 146)")
    parser.add_argument("--arg-atom", default="CZ", help="Atom name in Arg residue")
    parser.add_argument("--asp-atom", default="CG", help="Atom name in Asp residue")
    parser.add_argument("-o", "--output", default="distance_plot.png", help="Output image file (png or svg)")
    return parser.parse_args()

def compute_distances(pdb_file, chain_id, arg_res, asp_res, arg_atom_name, asp_atom_name):
    """
    Parse the PDB file and compute the distance between the specified atoms in each MODEL.
    Returns a list of model numbers and their corresponding distances.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AdK", pdb_file)
    distances = []
    model_ids = []

    for model in structure:
        try:
            # Get the chain from the model
            chain = model[chain_id]
            # Retrieve the specified residues
            arg_residue = chain[arg_res]
            asp_residue = chain[asp_res]

            # Get the specified atoms from each residue
            arg_atom = arg_residue[arg_atom_name]
            asp_atom = asp_residue[asp_atom_name]
            # Calculate the Euclidean distance between the two atoms
            distance = np.linalg.norm(arg_atom.get_coord() - asp_atom.get_coord())
            distances.append(distance)
            model_ids.append(model.id)
        except KeyError as e:
            # If the required atom or residue is missing in this model, print a warning and record NaN.
            print(f"Warning: Model {model.id} is missing required data: {e}")
            distances.append(np.nan)
            model_ids.append(model.id)
    return model_ids, distances

def plot_distances(model_ids, distances, output_file, arg_atom_name, asp_atom_name):
    """
    Plot the distance vs. model frame and save the figure.
    The plot labels and title are in English.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(model_ids, distances, marker='o', linestyle='-', color='blue')
    plt.xlabel("Model Frame Number", fontsize=12)
    plt.ylabel("Distance (Å)", fontsize=12)
    plt.title(f"Distance between Arg131 ({arg_atom_name}) and Asp146 ({asp_atom_name}) vs. Model Frame", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")

    # PPT-friendly version
    plt.figure(figsize=(8, 5))
    plt.plot(model_ids, distances, marker='o', linestyle='-', color='blue',
             linewidth=2.5, markersize=8)
    plt.xlabel("Model Frame Number", fontsize=16)
    plt.ylabel("Distance (Å)", fontsize=16)
    plt.title(f"Distance between Arg131 ({arg_atom_name}) and Asp146 ({asp_atom_name})", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=1.5)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    plt.tight_layout()
    output_base, output_ext = os.path.splitext(output_file)
    ppt_output_file = f"{output_base}_PPT{output_ext}"
    plt.savefig(ppt_output_file, dpi=300)
    print(f"PPT-friendly plot saved as {ppt_output_file}")

def main():
    args = parse_args()
    model_ids, distances = compute_distances(
        pdb_file=args.pdb_file,
        chain_id=args.chain,
        arg_res=args.arg_res,
        asp_res=args.asp_res,
        arg_atom_name=args.arg_atom,
        asp_atom_name=args.asp_atom
    )
    plot_distances(model_ids, distances, args.output, args.arg_atom, args.asp_atom)

if __name__ == "__main__":
    main()
