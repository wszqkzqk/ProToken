#!/usr/bin/env python3

import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.analysis import align
import argparse
import Xponge
# Force field modules might be needed if Xponge requires them for loading topology
import Xponge.forcefield.amber.ff14sb
import Xponge.forcefield.amber.tip3p
import Xponge.forcefield.amber.gaff

def load_and_select_calpha(pdb_path, atom_name="CA"):
    """Loads a PDB file using MDAnalysis and selects C-alpha atoms."""
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    try:
        u = mda.Universe(pdb_path)
        # Prioritize protein C-alphas, fallback if needed
        ag = u.select_atoms(f"name {atom_name} and protein")
        if len(ag) == 0:
             print(f"Warning: No atoms selected with 'name {atom_name} and protein'. Trying 'name {atom_name}' alone.")
             ag = u.select_atoms(f"name {atom_name}")
             if len(ag) == 0:
                 raise ValueError(f"No C-alpha atoms ('name {atom_name}') found in {pdb_path}.")
        print(f"Loaded {pdb_path} and found {len(ag)} C-alpha atoms.")
        # Return the AtomGroup containing only the selected C-alphas
        return u, ag
    except Exception as e:
        print(f"Error loading or selecting atoms from {pdb_path}: {e}")
        raise

def calculate_displacements(ag_start_ca, ag_target_ca):
    """Aligns start CAs to target CAs and calculates C-alpha displacements."""
    if len(ag_start_ca) != len(ag_target_ca):
        # Check if atom names/resids match to provide more insight
        start_ids = set((a.name, a.resid, a.resname) for a in ag_start_ca)
        target_ids = set((a.name, a.resid, a.resname) for a in ag_target_ca)
        diff = start_ids.symmetric_difference(target_ids)
        print(f"Warning: Mismatch in selected C-alpha atoms. Difference: {diff}")
        raise ValueError(f"Number of C-alpha atoms differs between start ({len(ag_start_ca)}) and target ({len(ag_target_ca)}). Ensure PDBs have corresponding CAs.")

    print("Aligning start structure C-alphas onto target structure C-alphas...")
    # Align the mobile CA group (start) onto the reference CA group (target)
    align.alignto(ag_start_ca, ag_target_ca, select="name CA", weights="mass") # Align only the CAs

    print("Calculating displacements...")
    # Get coordinates *after* alignment for start CAs, *original* for target CAs
    aligned_start_coords = ag_start_ca.positions
    original_target_coords = ag_target_ca.positions

    # Calculate displacement per C-alpha atom
    displacements = np.linalg.norm(aligned_start_coords - original_target_coords, axis=1)

    # Get the global 0-based indices *within the target Universe* for the target CAs
    target_global_indices = ag_target_ca.indices # These are 0-based indices

    # Return displacements aligned with the order of target_global_indices
    return displacements, target_global_indices

def select_key_atoms(displacements, target_global_indices, method="percentile", threshold=80.0):
    """Selects key atom indices based on displacement threshold."""
    if method == "percentile":
        if len(displacements) == 0:
              cutoff = 0.0
              print("Warning: No displacements calculated, percentile cutoff is 0.")
        else:
              cutoff = np.percentile(displacements, threshold)
        print(f"Using {threshold}th percentile displacement cutoff: {cutoff:.3f} Angstrom")
    elif method == "absolute":
        cutoff = threshold
        print(f"Using absolute displacement cutoff: {cutoff:.3f} Angstrom")
    else:
        raise ValueError("Invalid selection method. Choose 'percentile' or 'absolute'.")

    # Find *indices within the displacements array* where displacement > cutoff
    indices_meeting_criteria = np.where(displacements > cutoff)[0]

    if len(indices_meeting_criteria) == 0:
        print(f"Warning: No atoms selected with displacement > {cutoff:.3f}. Check structures or threshold.")
        # Optionally select top N if none meet criteria (adjust N as needed)
        if len(displacements) > 0:
             num_to_select = min(20, len(displacements)) # Select top 20 or fewer
             print(f"No atoms exceeded threshold. Selecting the top {num_to_select} most displaced atoms instead.")
             sorted_disp_indices = np.argsort(displacements)[::-1] # Indices sorted by displacement values (desc)
             indices_meeting_criteria = sorted_disp_indices[:num_to_select]
        else:
              print("Cannot select top atoms as there are no displacements.")
              return np.array([], dtype=int), cutoff # Return empty array


    # Get the *global 0-based indices* corresponding to these selected atoms
    # Use the indices_meeting_criteria to select from target_global_indices
    key_atom_global_indices = target_global_indices[indices_meeting_criteria] # These are 0-based indices


    print(f"Selected {len(key_atom_global_indices)} key C-alpha atoms based on displacement.")
    return key_atom_global_indices, cutoff


def extract_minimized_coords_and_write(
    topology_file_target,  # Topology corresponding to minimized coords (e.g., PDB or MOL2)
    minimized_coord_file_target, # SPONGE coord file (e.g., restart_coordinate.txt)
    key_atom_indices, # List/array of 0-based indices selected from displacement analysis
    output_target_coord_file,
    output_index_file,
    atom_name="CA" # No longer strictly needed here, but kept for consistency
):
    """
    Loads minimized target coordinates using Xponge and writes files for selected atoms
    using direct index access.

    ASSUMES atom order is IDENTICAL between the file used for MDAnalysis
    displacement calculation and the topology_file_target loaded here by Xponge.
    """
    print("-" * 20)
    print("Step 4a: Loading Target Topology and Minimized Coordinates using Xponge...")
    # --- Load topology and coordinates using Xponge ---
    if not os.path.exists(topology_file_target):
        raise FileNotFoundError(f"Target topology file not found: {topology_file_target}")
    if not os.path.exists(minimized_coord_file_target):
        raise FileNotFoundError(f"Minimized coordinate file not found: {minimized_coord_file_target}")

    print(f"Loading target topology from: {topology_file_target}...")
    try:
        if topology_file_target.lower().endswith(".pdb"):
            mol_target_minimized = Xponge.load_pdb(topology_file_target, ignore_hydrogen=True)
        elif topology_file_target.lower().endswith(".mol2"):
            # Ensure GAFF etc. are loaded if needed
            mol_target_minimized = Xponge.load_mol2(topology_file_target)
        else:
            raise ValueError("Unsupported target topology file format. Please use PDB or Mol2.")
        print(f"Target topology loaded successfully ({len(mol_target_minimized.atoms)} atoms).")
    except Exception as e:
        print(f"Error loading target topology with Xponge: {e}")
        raise

    print(f"Loading minimized coordinates from: {minimized_coord_file_target}...")
    try:
        # This function modifies atom.contents['x','y','z'] in mol_target_minimized
        Xponge.load_coordinate(minimized_coord_file_target, mol_target_minimized)
        print("Minimized coordinates loaded successfully into Xponge molecule.")
    except Exception as e:
        # Check atom count from coordinate file header if possible
        try:
            with open(minimized_coord_file_target, 'r') as cf:
                line1 = cf.readline().split()
                coord_natoms = int(line1[0])
            print(f"Coordinate file expects {coord_natoms} atoms.")
        except:
             pass # Ignore if reading header fails
        print(f"Error loading minimized coordinates with Xponge: {e}")
        print(f"Ensure atom count in topology ({len(mol_target_minimized.atoms)}) matches coordinate file.")
        raise
    # --- End of Xponge loading ---

    print("\nStep 4b: Extracting Coords/Indices for Key Atoms using Direct Indexing...")

    key_coords_list = []
    key_indices_output = []
    all_xponge_atoms = mol_target_minimized.atoms # Get the list of all Atom objects

    if not isinstance(key_atom_indices, (list, np.ndarray)):
         print("Warning: key_atom_indices is not a list or array. Converting.")
         key_atom_indices = list(key_atom_indices)

    if key_atom_indices.size == 0: # Check if the numpy array is not empty or all False if boolean
        print("Error: No valid indices provided for key atoms. Exiting.")
        exit(1)
    else:
        print(f"Processing {len(key_atom_indices)} selected key atom indices...")
        max_allowable_index = len(all_xponge_atoms) - 1
        valid_indices_count = 0
        for index in key_atom_indices:
            if 0 <= index <= max_allowable_index:
                # Access the atom directly using the 0-based index
                xponge_atom = all_xponge_atoms[index]

                # Extract coordinates from the atom's contents dictionary
                try:
                    coord = [
                        xponge_atom.contents['x'],
                        xponge_atom.contents['y'],
                        xponge_atom.contents['z']
                    ]
                    key_coords_list.append(coord)
                    key_indices_output.append(index)
                    valid_indices_count += 1
                except KeyError as ke:
                    print(f"Warning: Atom at index {index} is missing coordinate data ('{ke}'). Skipping.")
                except Exception as e_inner:
                     print(f"Warning: Error processing atom at index {index}: {e_inner}. Skipping.")

            else:
                print(f"Warning: Selected index {index} is out of bounds "
                      f"(molecule has {len(all_xponge_atoms)} atoms, "
                      f"max index {max_allowable_index}). Skipping.")

        print(f"Successfully extracted data for {valid_indices_count} key atoms.")


    key_coords = np.array(key_coords_list)

    # --- Write Output Files ---
    print(f"\nStep 4c: Writing SPONGE Input Files...")
    print(f"Writing coordinates of {len(key_coords)} key atoms to: {output_target_coord_file}")
    with open(output_target_coord_file, "w") as f:
        if key_coords.size > 0:
             np.savetxt(f, key_coords, fmt='%15.8f %15.8f %15.8f')
        else:
             f.write("")

    print(f"Writing indices of {len(key_indices_output)} key atoms to: {output_index_file}")
    with open(output_index_file, "w") as f:
        for index in key_indices_output:
            f.write(f"{index}\n")

    print("SPONGE input files generated successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPONGE SMD RMSD CV input files by selecting key C-alpha atoms based on displacement.")
    parser.add_argument("-s", "--start", required=True, help="Path to the starting state PDB file.")
    parser.add_argument("-e", "--target_pdb", required=True, help="Path to the target state PDB file. Used for displacement calculation.")
    parser.add_argument("-t", "--target_top", required=True, help="Path to the target state topology file compatible with Xponge and matching the minimized coordinates.")
    parser.add_argument("-c", "--target_min_coord", required=True, help="Path to the minimized coordinates file for the target state.")
    parser.add_argument("-a", "--atom_name", default="CA", help="Name of the C-alpha atom.")
    parser.add_argument("-m", "--selection_method", default="percentile", choices=["percentile", "absolute"], help="Method to select atoms: 'percentile' or 'absolute' cutoff (default: percentile).")
    parser.add_argument("-st", "--selection_threshold", default=80.0, type=float, help="Threshold for selection (percentile value 0-100 or absolute distance in Angstrom).")
    parser.add_argument("-o", "--out_dir", default=".", help="Output directory for generated files (default: current directory).")

    # Define default output filenames
    default_out_coord_filename = "target_calpha_ref.txt"
    default_out_index_filename = "rmsd_calpha_atoms.txt"

    args = parser.parse_args()

    # Resolve relative paths if necessary
    args.start = os.path.abspath(args.start)
    args.target_pdb = os.path.abspath(args.target_pdb)
    args.target_top = os.path.abspath(args.target_top)
    args.target_min_coord = os.path.abspath(args.target_min_coord)
    args.out_dir = os.path.abspath(args.out_dir) # Resolve output directory path

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory set to: {args.out_dir}")

    # Construct full output file paths
    output_coord_path = os.path.join(args.out_dir, default_out_coord_filename)
    output_index_path = os.path.join(args.out_dir, default_out_index_filename)

    print("--- Step 1: Loading Structures and Selecting C-alphas ---")
    u_start, ag_start_ca = load_and_select_calpha(args.start, args.atom_name)
    u_target, ag_target_ca = load_and_select_calpha(args.target_pdb, args.atom_name)

    print("\n--- Step 2: Calculating Displacements ---")
    # target_indices are the 0-based global indices from the target PDB
    displacements, target_indices = calculate_displacements(ag_start_ca, ag_target_ca)

    print("\n--- Step 3: Selecting Key Atoms ---")
    # key_atom_indices contains the selected 0-based global indices
    key_atom_indices, cutoff_value = select_key_atoms(
        displacements,
        target_indices,  # Pass the 0-based indices corresponding to the displacements
        method=args.selection_method,
        threshold=args.selection_threshold
    )

    print("\n--- Step 4: Extracting Minimized Coords and Writing Output ---")
    extract_minimized_coords_and_write(
        topology_file_target=args.target_top,
        minimized_coord_file_target=args.target_min_coord,
        key_atom_indices=key_atom_indices, # Use the selected 0-based indices
        output_target_coord_file=output_coord_path, # Use constructed path
        output_index_file=output_index_path,       # Use constructed path
        atom_name=args.atom_name
    )

    print("\nScript finished successfully.")
