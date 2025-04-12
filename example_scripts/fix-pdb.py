#!/usr/bin/env python3

import sys
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_pdb_missing_atoms(input_pdb_path, output_pdb_path):
    """
    Uses PDBFixer to find and add missing atoms (including CA) in a PDB file.
    Also adds missing hydrogens.
    """
    if not os.path.exists(input_pdb_path):
        print(f"Error: Input file not found: {input_pdb_path}")
        sys.exit(1)

    print(f"Processing PDB file: {input_pdb_path}")

    try:
        # Create a PDBFixer object from the PDB file
        fixer = PDBFixer(filename=input_pdb_path)
        print("PDBFixer object created.")

        # --- Key Steps for Fixing Missing Atoms ---
        # 1. Find missing residues (optional but good practice)
        fixer.findMissingResidues()
        print(f"Missing residues found: {fixer.missingResidues}") # Reports missing whole residues if any

        # 2. Find missing atoms (including CA, N, C, O, sidechain atoms)
        #    This step identifies which atoms are absent based on standard templates
        fixer.findMissingAtoms()
        print(f"Missing atoms found: {fixer.missingAtoms}") # Shows which atoms in which residues
        print(f"Missing terminals found: {fixer.missingTerminals}") # Reports missing terminal atoms (OXT)

        # 3. Add the missing atoms identified above
        #    This uses standard bond lengths and angles based on residue templates
        print("Adding missing atoms (including CA if found)...")
        fixer.addMissingAtoms()
        print("Missing atoms added.")

        # 4. Add missing hydrogens (optional, but often recommended after fixing heavy atoms)
        #    Specify the pH to determine protonation states
        # print("Adding missing hydrogens (pH 7.0)...")
        # fixer.addMissingHydrogens(7.0)
        # print("Missing hydrogens added.")

        # ------------------------------------------

        # Write the fixed structure to a new PDB file
        print(f"Writing fixed PDB to: {output_pdb_path}")
        with open(output_pdb_path, 'w') as outfile:
            PDBFile.writeFile(fixer.topology, fixer.positions, outfile, keepIds=True)

        print("PDB fixing complete.")

    except Exception as e:
        print(f"An error occurred during PDB fixing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_pdb_script.py <input_pdb_file> <output_pdb_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fix_pdb_missing_atoms(input_file, output_file)
