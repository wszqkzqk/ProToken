import argparse
import csv
import numpy as np
from Bio.PDB import PDBParser
import os
import math

# --- Configuration: Residue definitions from Li et al., 2021 ---
# N-domain: residues 108–231, and residues 269–271
DOMAIN_N_RANGES_STR = "108-231, 269-271"
# C-domain: residues 1–100, and residues 236–259
DOMAIN_C_RANGES_STR = "1-100, 236-259"
# Hinge segments: residues 101–107, 232–235 and 260–268
HINGE_RANGES_STR = "101-107, 232-235, 260-268"

# Default chain ID for RBP (e.g., 1URP, 2DRI have chain A)
DEFAULT_CHAIN_ID = 'A'

def parse_residue_ranges(range_str):
    """Parses a string like "1-5, 8, 10-12" into a set of residue numbers."""
    residue_ids = set()
    if not range_str:
        return residue_ids
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            for i in range(start, end + 1):
                residue_ids.add(i)
        else:
            residue_ids.add(int(part))
    return residue_ids

def get_ca_coords(structure, chain_id, target_residue_ids):
    """Extracts C-alpha coordinates for specified residues in a given chain."""
    coords = []
    model = structure[0] # Assuming first model
    if chain_id not in model:
        print(f"Warning: Chain {chain_id} not found in {structure.id}. Skipping.")
        return None # Or try to find any chain if only one exists

    chain = model[chain_id]
    for residue in chain:
        # res.id[0] is ' ' for standard residues, 'H_XXX' for hetatms
        if residue.id[0] == ' ' and residue.id[1] in target_residue_ids:
            if 'CA' in residue:
                coords.append(residue['CA'].get_coord())
            else:
                print(f"Warning: CA atom not found in residue {residue.id[1]} of chain {chain_id} in {structure.id}")
    
    if not coords:
        print(f"Warning: No CA atoms found for the specified residues in chain {chain_id} of {structure.id}")
        return None
        
    return np.array(coords)

def calculate_com(coords):
    """Calculates the center of mass for a list of coordinates."""
    if coords is None or len(coords) == 0:
        return None
    return np.mean(coords, axis=0)

def calculate_vector_angle(v1, v2):
    """Calculates the angle in degrees between two vectors."""
    if v1 is None or v2 is None:
        return None
    # Ensure they are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Check for zero vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Warning: One or both vectors are zero vectors. Cannot calculate angle.")
        return None

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clamp dot_product to avoid domain errors with acos due to floating point inaccuracies
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def process_pdb_file(pdb_path, d_n_res_ids, d_c_res_ids, h_res_ids, chain_id=DEFAULT_CHAIN_ID):
    """
    Processes a single PDB file to calculate the opening angle.
    Angle is defined by: COM_Hinge -> COM_Domain_N and COM_Hinge -> COM_Domain_C
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    except Exception as e:
        print(f"Error parsing PDB file {pdb_path}: {e}")
        return None

    # Get CA coordinates for each part
    domain_n_coords = get_ca_coords(structure, chain_id, d_n_res_ids)
    domain_c_coords = get_ca_coords(structure, chain_id, d_c_res_ids)
    hinge_coords = get_ca_coords(structure, chain_id, h_res_ids)

    if domain_n_coords is None or domain_c_coords is None or hinge_coords is None:
        print(f"Could not get coordinates for all domains/hinge in {pdb_path}. Skipping angle calculation.")
        return None
    if len(domain_n_coords) == 0 or len(domain_c_coords) == 0 or len(hinge_coords) == 0:
        print(f"One of the domains or hinge region has no C-alpha atoms in {pdb_path}. Skipping angle calculation.")
        return None

    # Calculate COMs
    com_domain_n = calculate_com(domain_n_coords)
    com_domain_c = calculate_com(domain_c_coords)
    com_hinge = calculate_com(hinge_coords)

    if com_domain_n is None or com_domain_c is None or com_hinge is None:
        print(f"Could not calculate COM for all domains/hinge in {pdb_path}. Skipping angle calculation.")
        return None

    # Create vectors from Hinge COM to Domain COMs
    vec_hinge_to_N = com_domain_n - com_hinge
    vec_hinge_to_C = com_domain_c - com_hinge
    
    # Calculate angle
    angle = calculate_vector_angle(vec_hinge_to_N, vec_hinge_to_C)
    return angle

def main():
    parser = argparse.ArgumentParser(description="Calculate RBP opening angle from PDB files based on domain and hinge COMs.")
    parser.add_argument("pdb_files", nargs='+', help="Paths to one or more PDB files.")
    parser.add_argument("--chain_id", default=DEFAULT_CHAIN_ID, help=f"Chain ID to use for RBP (default: {DEFAULT_CHAIN_ID}).")
    parser.add_argument("--output_csv", default="rbp_opening_angles.csv", help="Output CSV file name.")
    
    args = parser.parse_args()

    domain_n_ids = parse_residue_ranges(DOMAIN_N_RANGES_STR)
    domain_c_ids = parse_residue_ranges(DOMAIN_C_RANGES_STR)
    hinge_ids = parse_residue_ranges(HINGE_RANGES_STR)

    if not domain_n_ids or not domain_c_ids or not hinge_ids:
        print("Error: Could not parse domain/hinge residue ranges. Check definitions.")
        return

    results = []
    for pdb_file in args.pdb_files:
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file {pdb_file} not found. Skipping.")
            continue
        print(f"Processing {pdb_file}...")
        angle = process_pdb_file(pdb_file, domain_n_ids, domain_c_ids, hinge_ids, args.chain_id)
        if angle is not None:
            results.append({"PDB File": os.path.basename(pdb_file), "Opening Angle (degrees)": f"{angle:.2f}"})
        else:
            results.append({"PDB File": os.path.basename(pdb_file), "Opening Angle (degrees)": "Error"})


    if results:
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = ["PDB File", "Opening Angle (degrees)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nResults saved to {args.output_csv}")
    else:
        print("No PDB files processed or no valid angles calculated.")

if __name__ == "__main__":
    main()
