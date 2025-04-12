#!/usr/bin/env python3

import MDAnalysis as mda
# Remove RMSD import, add tmtools
# from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import warnings
from tqdm import tqdm
import tmtools # Added import

# Suppress PDB reading warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.topology.PDBParser')

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein trajectory path using TM-score.", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("trajectory", help="Path to the multi-frame PDB trajectory file.")
    parser.add_argument("ref_start", help="Path to the reference Start PDB file.")
    parser.add_argument("ref_end", help="Path to the reference End PDB file.")
    parser.add_argument(
        "--select",
        default="protein and name CA", # Changed default to CA atoms, common for TM-score
        help="MDAnalysis selection string for atoms (e.g., 'protein and name CA')."
    )
    # Updated threshold descriptions and defaults for TM-score (0-1 range)
    parser.add_argument(
        "--deviation_threshold",
        type=float,
        default=0.5, # TM-score threshold, lower means more deviation
        help="Absolute TM-score threshold below which a frame is considered deviating from BOTH ends."
    )
    parser.add_argument(
        "--smoothness_threshold",
        type=float,
        default=0.9, # Frame-to-frame TM-score threshold, lower means less smooth
        help="Frame-to-frame TM-score threshold below which a potential discontinuity is flagged."
         )
    # Updated output prefix default
    parser.add_argument("--output_prefix", default="path_tmscore_eval", help="Prefix for output plot files and data.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip generating plots.")

    args = parser.parse_args()

    # --- Load Structures ---
    print("Loading structures...")

    # Set output directory based on trajectory file location
    trajectory_dir = os.path.dirname(os.path.abspath(args.trajectory))
    output_dir = trajectory_dir if not os.path.isabs(args.output_prefix) else os.path.dirname(args.output_prefix)
    output_prefix = os.path.join(output_dir, os.path.basename(args.output_prefix))

    try:
        traj_u = mda.Universe(args.trajectory)
        start_u = mda.Universe(args.ref_start)
        end_u = mda.Universe(args.ref_end)
    except Exception as e:
        print(f"Error loading PDB files: {e}")
        return

    # --- Select Atoms and Check Consistency ---
    try:
        atom_selection = args.select # Renamed variable for clarity
        traj_atoms = traj_u.select_atoms(atom_selection)
        start_atoms = start_u.select_atoms(atom_selection)
        end_atoms = end_u.select_atoms(atom_selection)

        # Check atom counts
        n_atoms_traj = len(traj_atoms)
        n_atoms_start = len(start_atoms)
        n_atoms_end = len(end_atoms)

        if not (n_atoms_traj > 0 and n_atoms_start > 0 and n_atoms_end > 0):
             print(f"Error: Selection '{atom_selection}' resulted in 0 atoms in one or more structures.")
             return
        # TM-score requires sequences, check if lengths match if selection differs (though usually it should be the same)
        # For simplicity, we'll assume the selection yields the same number of atoms if the selection string is identical.
        # A more robust check would compare residue sequences if necessary.
        if not (n_atoms_traj == n_atoms_start == n_atoms_end):
            print(f"Warning: Atom count mismatch in selection '{atom_selection}'.")
            print(f"Trajectory: {n_atoms_traj}, Start: {n_atoms_start}, End: {n_atoms_end}")
            print("Attempting to proceed, but TM-score might be unreliable if sequences differ significantly.")
            # Depending on tmtools version, it might handle slight mismatches or require exact matches.
            # Let's assume for now the user ensures the selection is comparable.

        print(f"Using selection '{atom_selection}' resulting in approximately {n_atoms_traj} atoms per frame.")

        # Extract sequences (required by tmtools) - Use CA atoms for sequence extraction
        # Assuming standard residues; adjust if using non-standard ones or specific atom types
        try:
            seq_start = start_atoms.residues.sequence(id_type='resname', format='string')
            seq_end = end_atoms.residues.sequence(id_type='resname', format='string')
            # Get sequence from the first frame of the trajectory
            traj_atoms_frame0 = traj_u.select_atoms(atom_selection) # Re-select for frame 0
            traj_u.trajectory[0] # Go to first frame
            seq_traj = traj_atoms_frame0.residues.sequence(id_type='resname', format='string')

            if not (len(seq_start) == len(seq_end) == len(seq_traj)):
                 print("Warning: Sequence length mismatch detected based on selection!")
                 print(f"Start: {len(seq_start)}, End: {len(seq_end)}, Traj: {len(seq_traj)}")
                 # Decide how to handle: error out or proceed with caution? Let's warn and proceed.
                 # tmtools might handle this internally or raise an error later.
            # Use the start sequence as the reference sequence length for normalization if needed
            ref_len = len(seq_start)

        except Exception as e:
            print(f"Error extracting sequence information: {e}. Ensure selection includes standard protein residues.")
            return

    except mda.SelectionError as e:
        print(f"Error during atom selection: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
        return


    # --- Calculate TM-score(Start, End) ---
    start_ref_coords = start_atoms.positions
    end_ref_coords = end_atoms.positions

    # Use tm_align to get TM-score between start and end
    # Normalization is handled internally by tmtools based on sequence length
    tm_results_se = tmtools.tm_align(start_ref_coords, end_ref_coords, seq_start, seq_end)
    # tm_results_se.tm_norm_chain1 uses len(seq_start), tm_results_se.tm_norm_chain2 uses len(seq_end)
    # If lengths are equal, these are the same. Let's use chain1 normalization consistently.
    tmscore_start_end = tm_results_se.tm_norm_chain1
    print(f"TM-score between Start and End structures: {tmscore_start_end:.4f}")
    # Deviation threshold is now absolute
    deviation_tmscore_threshold_value = args.deviation_threshold
    print(f"Deviation TM-score threshold: {deviation_tmscore_threshold_value:.4f} (frames below this are deviating)")
    print(f"Smoothness TM-score threshold: {args.smoothness_threshold:.4f} (frame-pairs below this are discontinuous)")


    # --- Process Trajectory ---
    print(f"Processing {len(traj_u.trajectory)} frames...")
    results = {
        "frame": [],
        "tmscore_to_start": [], # Renamed
        "tmscore_to_end": [],   # Renamed
        "tmscore_frame_to_frame": [], # Renamed
        "is_deviating": [],
        "is_discontinuous": []
    }

    prev_frame_coords = None
    prev_frame_seq = None # Need sequence for frame-to-frame

    # Re-select atoms within the loop to ensure coordinates are updated
    mobile_atoms = traj_u.select_atoms(atom_selection)

    for i, ts in enumerate(tqdm(traj_u.trajectory)):
        results["frame"].append(i)

        current_coords = mobile_atoms.positions # Get current coordinates
        # Extract sequence for the current frame (can be slow if selection is complex)
        # Optimization: If sequence is known to be constant, extract only once outside the loop.
        # Assuming sequence *could* change if the selection logic depends on frame-specific properties (unlikely for 'name CA').
        current_seq = mobile_atoms.residues.sequence(id_type='resname', format='string')
        if len(current_seq) != ref_len:
             print(f"Warning: Sequence length changed at frame {i}. This might affect TM-score consistency.")
             # Handle error or adapt logic if needed

        # 1. TM-score to References
        tm_results_s = tmtools.tm_align(current_coords, start_ref_coords, current_seq, seq_start)
        tm_results_e = tmtools.tm_align(current_coords, end_ref_coords, current_seq, seq_end)
        # Normalize TM-score by the length of the reference structure (start/end)
        tmscore_s = tm_results_s.tm_norm_chain2 # Normalized by len(seq_start)
        tmscore_e = tm_results_e.tm_norm_chain2 # Normalized by len(seq_end)
        results["tmscore_to_start"].append(tmscore_s)
        results["tmscore_to_end"].append(tmscore_e)

        # 2. Path Deviation Check (Low TM-score to BOTH ends)
        is_deviating = (tmscore_s < deviation_tmscore_threshold_value and
                        tmscore_e < deviation_tmscore_threshold_value)
        results["is_deviating"].append(is_deviating)

        # 3. Frame-to-Frame TM-score (Path Smoothness)
        if prev_frame_coords is not None:
            # Calculate TM-score between consecutive frames
            tm_results_f2f = tmtools.tm_align(current_coords, prev_frame_coords, current_seq, prev_frame_seq)
            # Normalize by the length of the *first* structure in the comparison (current frame)
            tmscore_f2f = tm_results_f2f.tm_norm_chain1
            results["tmscore_frame_to_frame"].append(tmscore_f2f)
            # Check if TM-score is below the smoothness threshold
            is_discontinuous = tmscore_f2f < args.smoothness_threshold
        else:
            results["tmscore_frame_to_frame"].append(1.0) # First frame compared to nothing is perfectly similar (TM=1)
            is_discontinuous = False # Cannot be discontinuous
        results["is_discontinuous"].append(is_discontinuous)

        # Store current coordinates and sequence for the next iteration
        prev_frame_coords = current_coords.copy() # Make a copy
        prev_frame_seq = current_seq

    # --- Analysis Summary ---
    print("\n--- Path TM-score Analysis Summary ---") # Updated title
    results_np = {k: np.array(v) for k, v in results.items()} # Convert to numpy arrays

    # Calculate min frame-to-frame TM-score (excluding the first frame's placeholder)
    min_f2f_tmscore = results_np['tmscore_frame_to_frame'][1:].min() if len(results['frame']) > 1 else 1.0
    print(f"Min Frame-to-Frame TM-score: {min_f2f_tmscore:.4f}")
    n_discontinuous = results_np['is_discontinuous'][1:].sum() # Skip first frame's False value
    print(f"Frames with potential discontinuity (Frame-to-Frame TM-score < {args.smoothness_threshold:.2f}): {n_discontinuous} ({n_discontinuous/max(1, len(results['frame'])-1)*100:.1f}%)")

    n_deviating = results_np['is_deviating'].sum()
    print(f"Frames marked as 'deviating' (TM-score to both ends < {deviation_tmscore_threshold_value:.2f}): {n_deviating} ({n_deviating/len(results['frame'])*100:.1f}%)")
    if n_deviating > 0:
        deviating_indices = results_np['frame'][results_np['is_deviating']]
        print(f"  Deviating frame indices: {deviating_indices[:10]}..." if len(deviating_indices) > 10 else deviating_indices)
    if n_discontinuous > 0:
        # Find indices where discontinuity occurs (index i means transition from i-1 to i was discontinuous)
        discontinuous_indices = results_np['frame'][results_np['is_discontinuous']]
        print(f"  Discontinuous frame transition indices (frame i vs i-1): {discontinuous_indices[:10]}..." if len(discontinuous_indices) > 10 else discontinuous_indices)


    # --- Save Data ---
    data_filename = f"{output_prefix}_tmscore_metrics.csv" # Updated filename
    try:
        header = ",".join(results.keys())
        # Prepare data for saving
        data_out_list = []
        keys = list(results.keys())
        for i in range(len(results["frame"])):
             row = [results[k][i] for k in keys]
             data_out_list.append(row)
        data_out = np.array(data_out_list)

        # Format booleans as integers (0 or 1)
        bool_cols_indices = [keys.index(k) for k in ['is_deviating', 'is_discontinuous']]
        data_out[:, bool_cols_indices] = data_out[:, bool_cols_indices].astype(int)

        # Update format string for TM-scores (floats)
        np.savetxt(data_filename, data_out, delimiter=",", header=header, fmt=['%d'] + ['%.6f'] * 3 + ['%d'] * 2, comments='') # Use %.6f for TM-scores
        print(f"Per-frame TM-score metrics saved to: {data_filename}")
    except Exception as e:
        print(f"Warning: Could not save metrics data to CSV. {e}")

    # --- Plotting ---
    if not args.skip_plots:
        print("Generating plots...")
        frames = results_np['frame']

        # Plot 1: Frame-to-Frame TM-score (Path Smoothness)
        fig_smooth, ax_smooth = plt.subplots(figsize=(10, 4))
        # Plot TM-score, skip the first frame's placeholder value
        ax_smooth.plot(frames[1:], results_np['tmscore_frame_to_frame'][1:], label='TM-score', color='purple', linewidth=1)
        # Add threshold line
        # ax_smooth.axhline(args.smoothness_threshold, color='r', linestyle='--', label=f'Threshold ({args.smoothness_threshold:.2f})')
        ax_smooth.set_xlabel('Frame Index (i vs i-1)')
        ax_smooth.set_ylabel('Frame-to-Frame TM-score') # Updated label
        ax_smooth.set_title('Path Smoothness Analysis (Frame-to-Frame TM-score)') # Updated title
        ax_smooth.set_ylim(0, 1.05) # TM-score range is 0-1
        ax_smooth.legend()
        ax_smooth.grid(True, alpha=0.5)
        fig_smooth.tight_layout()
        smooth_plot_filename = f"{output_prefix}_tmscore_smoothness_plot.png" # Updated filename
        fig_smooth.savefig(smooth_plot_filename, dpi=150)
        print(f"Smoothness plot saved to: {smooth_plot_filename}")

        # Plot 2: Deviation TM-score Scatter Plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 7))
        scatter_colors = frames
        # Plot TM-score to start vs TM-score to end
        sc = ax_scatter.scatter(results_np['tmscore_to_start'], results_np['tmscore_to_end'],
                                c=scatter_colors, cmap='viridis', s=10, alpha=0.7,
                                label='Frames (color=Frame Index)')
        # Highlight deviating points (low TM-score to both)
        deviating_mask = results_np['is_deviating']
        ax_scatter.scatter(results_np['tmscore_to_start'][deviating_mask], results_np['tmscore_to_end'][deviating_mask],
                           facecolors='none', edgecolors='red', s=30, label=f'Deviating (Both TM < {deviation_tmscore_threshold_value:.2f})') # Updated label

        ax_scatter.set_xlabel(f"TM-score to Start ({os.path.basename(args.ref_start)})") # Updated label
        ax_scatter.set_ylabel(f"TM-score to End ({os.path.basename(args.ref_end)})")   # Updated label
        ax_scatter.set_title('Path Deviation Analysis (TM-score)') # Updated title
        # ax_scatter.set_xlim(0, 1.05) # TM-score range
        # ax_scatter.set_ylim(0, 1.05) # TM-score range
        if deviating_mask.any():
            # Draw threshold lines for clarity
            ax_scatter.axhline(deviation_tmscore_threshold_value, color='grey', linestyle=':', alpha=0.7)
            ax_scatter.axvline(deviation_tmscore_threshold_value, color='grey', linestyle=':', alpha=0.7, label=f'Deviation Threshold ({deviation_tmscore_threshold_value:.2f})') # Updated label
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(fontsize='small')
        cbar = fig_scatter.colorbar(sc)
        cbar.set_label('Frame Index')
        fig_scatter.tight_layout()
        scatter_plot_filename = f"{output_prefix}_tmscore_deviation_scatter.png" # Updated filename
        fig_scatter.savefig(scatter_plot_filename, dpi=150)
        print(f"Deviation scatter plot saved to: {scatter_plot_filename}")

    print("\nPath TM-score evaluation complete.") # Updated message


if __name__ == "__main__":
    main()
