#!/usr/bin/env python3

import argparse
import os
import re
from glob import glob

def natural_sort_key(s):
    """Natural sorting (numeric order instead of lexicographic)"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def merge_pdbs(input_dir, output_pdb):
    # Collect and sort PDB files
    pdb_files = sorted(
        glob(os.path.join(input_dir, "*.pdb")),
        key=lambda x: natural_sort_key(os.path.basename(x))
    )
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {input_dir}")

    with open(output_pdb, 'w') as outfile:
        for idx, pdb_file in enumerate(pdb_files, 1):
            # Write MODEL header
            outfile.write(f"MODEL        {idx}\n")
            with open(pdb_file, 'r') as infile:
                # Copy content line by line (skip existing MODEL/ENDMDL)
                for line in infile:
                    if not line.startswith(("MODEL", "ENDMDL")):
                        outfile.write(line)
            # Write ENDMDL footer
            outfile.write("\nENDMDL\n")
    print(f"Merged {len(pdb_files)} frames -> {output_pdb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple PDB files into multi-model PDB trajectory (NMR-style)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Input directory containing PDB files (supports *.pdb)"
    )
    parser.add_argument(
        "-o", "--output_pdb",
        required=True,
        help="Output multi-model PDB file path (e.g., merged.pdb)"
    )
    args = parser.parse_args()

    try:
        merge_pdbs(args.input_dir, args.output_pdb)
    except Exception as e:
        print(f"Error: {str(e)}")
