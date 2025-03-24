#!/usr/bin/env python3

import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute RMSD of a multi-model PDB file relative to a reference structure using Cα atoms."
    )
    parser.add_argument("pdb_file", help="Input multi-model PDB file (e.g., 100-frame simulation)")
    parser.add_argument("ref_pdb", help="Reference PDB file (e.g., 4AKE or 1AKE)")
    parser.add_argument("-o", "--output", default="rmsd_values.txt", help="Output file to save RMSD values")
    return parser.parse_args()

def compute_rmsd(pdb_file, ref_pdb):
    """
    Load the trajectory and reference structure, then compute the Cα RMSD for each frame.
    """
    # Load the reference structure and the trajectory
    u = mda.Universe(ref_pdb, pdb_file)
    ref = mda.Universe(ref_pdb)
    
    # Select Cα atoms for reference and for each frame in the trajectory
    ref_ca = ref.select_atoms("name CA")
    mobile_ca = u.select_atoms("name CA")
    
    rmsd_list = []
    # Iterate over each frame in the trajectory
    for ts in u.trajectory:
        # Compute RMSD after centering and superposition
        rmsd_value = rms.rmsd(mobile_ca.positions, ref_ca.positions, center=True, superposition=True)
        rmsd_list.append(rmsd_value)
    return rmsd_list

def main():
    args = parse_args()
    rmsd_values = compute_rmsd(args.pdb_file, args.ref_pdb)
    
    # Save RMSD values to an output file
    with open(args.output, "w") as f:
        for i, value in enumerate(rmsd_values, start=1):
            f.write(f"{i}\t{value}\n")
    print(f"RMSD values saved to {args.output}")

if __name__ == "__main__":
    main()
