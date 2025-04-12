#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import tmtools # Use tmtools for TM-score calculation
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm

# Suppress PDB reading warnings from MDAnalysis for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.topology.PDBParser')

def get_coords_and_sequence(universe, selection="protein and name CA"):
    """Extracts coordinates and sequence for selected atoms."""
    ag = universe.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection '{selection}' resulted in 0 atoms.")
    coords = ag.positions
    # Attempt to get sequence - handle potential issues
    try:
        # Assuming standard residues and protein for sequence extraction
        protein_ag = ag.residues.atoms.select_atoms("protein")
        sequence = "".join(protein_ag.residues.sequence())
        # Basic check if sequence length roughly matches atom group length if CA selected
        if "name CA" in selection.lower() and len(sequence) != len(ag):
             print(f"Warning: Sequence length ({len(sequence)}) doesn't match selected CA atoms ({len(ag)}). Using sequence derived from selected residues.")
             # Fallback: derive sequence directly from the selected residues if possible
             try:
                 sequence = "".join(ag.residues.sequence())
                 if len(sequence) != len(ag.residues): # Check if residue count matches
                     print(f"Warning: Fallback sequence length ({len(sequence)}) still mismatching residue count ({len(ag.residues)}). TM-score might be affected.")
             except Exception as e:
                 print(f"Error getting fallback sequence: {e}. Cannot proceed with TM-score.")
                 return None, None

        if not sequence: # If sequence is empty
             raise ValueError("Could not extract a valid sequence for TM-score calculation.")

    except Exception as e:
        print(f"Warning: Could not automatically determine sequence for selection '{selection}'. Error: {e}. TM-score calculation might fail or be inaccurate. Ensure input PDBs have standard residue information.")
        # Cannot reliably calculate TM-score without sequence
        return None, None # Indicate failure

    return coords, sequence


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein conformation sampling against two reference structures using RMSD and TM-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("trajectory", help="Path to the multi-frame PDB trajectory file.")
    parser.add_argument("ref1", help="Path to the first reference PDB file.")
    parser.add_argument("ref2", help="Path to the second reference PDB file.")
    parser.add_argument(
        "--select",
        default="protein and name CA",
        help="MDAnalysis selection string for atoms to use in RMSD and TM-score calculations."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="TM-score threshold for successful sampling criterion."
    )

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file not found: {args.trajectory}")
        return
    if not os.path.exists(args.ref1):
        print(f"Error: Reference 1 file not found: {args.ref1}")
        return
    if not os.path.exists(args.ref2):
        print(f"Error: Reference 2 file not found: {args.ref2}")
        return

    # --- Load Structures ---
    print(f"Loading reference 1: {args.ref1}")
    try:
        ref1_u = mda.Universe(args.ref1)
        ref1_coords, ref1_seq = get_coords_and_sequence(ref1_u, args.select)
        if ref1_coords is None: return # Error handled in function
        ref1_selection_group = ref1_u.select_atoms(args.select)
        n_atoms_ref1 = len(ref1_selection_group)
        print(f"Reference 1: Selected {n_atoms_ref1} atoms with selection '{args.select}'")
    except Exception as e:
        print(f"Error loading or processing reference 1: {e}")
        return

    print(f"Loading reference 2: {args.ref2}")
    try:
        ref2_u = mda.Universe(args.ref2)
        ref2_coords, ref2_seq = get_coords_and_sequence(ref2_u, args.select)
        if ref2_coords is None: return # Error handled in function
        ref2_selection_group = ref2_u.select_atoms(args.select)
        n_atoms_ref2 = len(ref2_selection_group)
        print(f"Reference 2: Selected {n_atoms_ref2} atoms with selection '{args.select}'")
    except Exception as e:
        print(f"Error loading or processing reference 2: {e}")
        return

    # --- Sequence/Atom Count Consistency Check ---
    if n_atoms_ref1 != n_atoms_ref2:
        print(f"Error: Number of selected atoms differs between references ({n_atoms_ref1} vs {n_atoms_ref2}) using selection '{args.select}'. Cannot compare.")
        return
    if ref1_seq != ref2_seq:
        # This might be okay if atom selection is consistent, but TM-score needs sequence
        print(f"Warning: Sequences derived from references differ. Ensure selection '{args.select}' is valid for both structures. TM-score will use respective sequences.")
        # If sequences differ significantly, TM-score comparison might be less meaningful unless intended.

    n_atoms_ref = n_atoms_ref1 # Use this for checking trajectory frames

    print(f"Loading trajectory: {args.trajectory}")
    try:
        traj_u = mda.Universe(args.trajectory)
        # Check first frame atom count consistency
        traj_selection_group_check = traj_u.select_atoms(args.select)
        if len(traj_selection_group_check) != n_atoms_ref:
             print(f"Error: Number of selected atoms in trajectory ({len(traj_selection_group_check)}) does not match references ({n_atoms_ref}) using selection '{args.select}'.")
             return
        # Get sequence from trajectory (assuming it's consistent across frames)
        _, traj_seq = get_coords_and_sequence(traj_u, args.select)
        if traj_seq is None:
            print("Error: Could not get sequence from trajectory. Cannot calculate TM-score.")
            return
        # Check sequence consistency with references (important for TM-score)
        if traj_seq != ref1_seq:
             print(f"Warning: Trajectory sequence differs from reference 1 sequence. TM-score comparison might be affected.")
        if traj_seq != ref2_seq:
             print(f"Warning: Trajectory sequence differs from reference 2 sequence. TM-score comparison might be affected.")

    except Exception as e:
        print(f"Error loading or processing trajectory: {e}")
        return

    # --- Process Trajectory Frames ---
    tm_scores_ref1 = []
    tm_scores_ref2 = []
    rmsds_ref1 = []
    rmsds_ref2 = []

    print(f"\nProcessing {len(traj_u.trajectory)} frames...")
    mobile_selection_group = traj_u.select_atoms(args.select) # Select once outside loop

    for i, ts in enumerate(tqdm(traj_u.trajectory, desc="Frames")):
        try:
            # Ensure the selection group is updated for the current frame implicitly by MDA
            # Get current frame coordinates for the selection
            mobile_coords = mobile_selection_group.positions.copy() # Use copy to avoid modification by align

            # --- Reference 1 Comparison ---
            # TM-score (tmtools handles its own alignment)
            # Use sequences derived from *each* structure for tm_align
            tm_results_ref1 = tmtools.tm_align(mobile_coords, ref1_coords, traj_seq, ref1_seq)
            tm_scores_ref1.append(tm_results_ref1.tm_norm_chain1) # Score normalized by length of chain1 (mobile)

            # RMSD (requires pre-alignment)
            # Align mobile (current frame selection) to ref1_selection_group
            # alignment happens in-place on traj_u's coordinates for the selection
            align.alignto(mobile_selection_group, ref1_selection_group, select=args.select, weights="mass")
            # Calculate RMSD on the now-aligned coordinates
            current_rmsd_ref1 = rms.rmsd(mobile_selection_group.positions, ref1_selection_group.positions, superposition=False) # superposition=False because alignto already did it
            rmsds_ref1.append(current_rmsd_ref1)


            # --- Reference 2 Comparison ---
             # Use the original mobile_coords saved earlier, as alignment modified traj_u
            mobile_coords_orig = mobile_coords # Reuse the copy taken at the start of the loop

            # TM-score
            tm_results_ref2 = tmtools.tm_align(mobile_coords_orig, ref2_coords, traj_seq, ref2_seq)
            tm_scores_ref2.append(tm_results_ref2.tm_norm_chain1)

            # RMSD (requires pre-alignment to Ref 2)
            # Reset trajectory coords implicitly by moving to next frame or reload if needed
            # For RMSD, we need to align *this frame* to *ref 2*.
            # Since alignto works in-place, we need to ensure we are aligning the *original* state of this frame to ref2.
            # Best practice: reload frame or realign from a saved state if needed.
            # Simpler approach if alignto doesn't mess up *other* atoms: just realign
            # Let's re-select and align to be safe (though slightly less efficient)
            traj_u.trajectory[i] # Ensure we are on the correct frame
            mobile_selection_group_reselect = traj_u.select_atoms(args.select) # Reselect
            align.alignto(mobile_selection_group_reselect, ref2_selection_group, select=args.select, weights="mass")
            current_rmsd_ref2 = rms.rmsd(mobile_selection_group_reselect.positions, ref2_selection_group.positions, superposition=False)
            rmsds_ref2.append(current_rmsd_ref2)


        except Exception as e:
            print(f"\nWarning: Error processing frame {i}. Skipping. Error: {e}")
            # Add NaN or skip results for this frame if needed, here we just skip
            continue # Skip to next frame


    # --- Aggregate Results ---
    if not tm_scores_ref1 or not rmsds_ref1:
        print("\nError: No frames were successfully processed.")
        return

    tm_scores_ref1_np = np.array(tm_scores_ref1)
    tm_scores_ref2_np = np.array(tm_scores_ref2)
    rmsds_ref1_np = np.array(rmsds_ref1)
    rmsds_ref2_np = np.array(rmsds_ref2)

    best_tm_ref1 = np.max(tm_scores_ref1_np)
    best_tm_ref2 = np.max(tm_scores_ref2_np)
    best_rmsd_ref1 = np.min(rmsds_ref1_np)
    best_rmsd_ref2 = np.min(rmsds_ref2_np)

    # --- Evaluate Success ---
    # Based on paper: "successful sampling" to be that each of the best TM scores
    # for a given case is larger than 0.90 (Table 1 description)
    is_successful = (best_tm_ref1 >= args.threshold) and (best_tm_ref2 >= args.threshold)

    # --- Output Results ---
    print("\n--- Evaluation Results ---")
    print(f"Processed {len(tm_scores_ref1)} frames successfully.")
    print(f"Selection used: '{args.select}'")
    print("-" * 25)
    print(f"Best TM-score to Reference 1 ({os.path.basename(args.ref1)}): {best_tm_ref1:.4f}")
    print(f"Best TM-score to Reference 2 ({os.path.basename(args.ref2)}): {best_tm_ref2:.4f}")
    print("-" * 25)
    print(f"Best (minimum) RMSD to Reference 1: {best_rmsd_ref1:.4f} Å")
    print(f"Best (minimum) RMSD to Reference 2: {best_rmsd_ref2:.4f} Å")
    print("-" * 25)
    print(f"Successful Sampling (Best TM > {args.threshold} for BOTH refs): {is_successful}")
    print("-" * 25)

if __name__ == "__main__":
    main()
