#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import warnings
from tqdm import tqdm

# Suppress PDB reading warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.topology.PDBParser')

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate backbone protein trajectory path smoothness and deviation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("trajectory", help="Path to the multi-frame PDB trajectory file.")
    parser.add_argument("ref_start", help="Path to the reference Start PDB file.")
    parser.add_argument("ref_end", help="Path to the reference End PDB file.")
    parser.add_argument(
        "--select",
        default="backbone or (protein and (name N CA C O))",
        help="MDAnalysis selection string for backbone atoms."
    )
    parser.add_argument(
        "--deviation_threshold_multiplier",
        type=float,
        default=1.0,
        help="RMSD multiplier relative to RMSD(Start,End) to flag deviating points."
    )
    parser.add_argument(
        "--smoothness_multiplier",
        type=float,
        default=50.0,
        help="Multiplier for smoothness threshold calculation: x * RMSD(Start,End) / numFrames"
    )
    parser.add_argument("--output_prefix", default="path_geom_eval", help="Prefix for output plot files and data.")
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
        traj_backbone_selection = args.select
        traj_backbone = traj_u.select_atoms(traj_backbone_selection)
        start_backbone = start_u.select_atoms(traj_backbone_selection)
        end_backbone = end_u.select_atoms(traj_backbone_selection)

        n_atoms_traj = len(traj_backbone)
        n_atoms_start = len(start_backbone)
        n_atoms_end = len(end_backbone)

        if not (n_atoms_traj == n_atoms_start == n_atoms_end):
            print(f"Error: Atom count mismatch in selection '{traj_backbone_selection}'!")
            print(f"Trajectory: {n_atoms_traj}, Start: {n_atoms_start}, End: {n_atoms_end}")
            return
        if n_atoms_traj == 0:
            print(f"Error: Selection '{traj_backbone_selection}' resulted in 0 atoms.")
            return
        print(f"Using {n_atoms_traj} backbone atoms per frame based on selection.")
    except Exception as e:
        print(f"Error during atom selection: {e}")
        return

    # --- Calculate RMSD(Start, End) ---
    # Align references first for a stable base RMSD calculation
    start_ref_coords = start_backbone.positions.copy()
    end_ref_coords = end_backbone.positions.copy()
    # Align end to start (doesn't matter which way for RMSD value)
    rms.rmsd(end_ref_coords, start_ref_coords, superposition=True) # Aligns end_ref_coords in place
    rmsd_start_end = rms.rmsd(start_ref_coords, end_ref_coords, superposition=True) # Calculate RMSD on aligned coords
    print(f"RMSD between aligned Start and End structures: {rmsd_start_end:.3f} Å")
    deviation_rmsd_threshold_value = args.deviation_threshold_multiplier * rmsd_start_end
    print(f"Deviation RMSD threshold: {deviation_rmsd_threshold_value:.3f} Å ({args.deviation_threshold_multiplier} * RMSDse)")
    num_frames = len(traj_u.trajectory)
    smoothness_threshold = args.smoothness_multiplier * rmsd_start_end / num_frames
    print(f"Smoothness threshold: {smoothness_threshold:.3f} Å ({args.smoothness_multiplier} * RMSDse / numFrames)")

    # --- Process Trajectory ---
    print(f"Processing {len(traj_u.trajectory)} frames...")
    results = {
        "frame": [],
        "rmsd_to_start": [],
        "rmsd_to_end": [],
        "rmsd_frame_to_frame": [],
        "is_deviating": [],
        "is_discontinuous": []
    }

    prev_frame_coords = None
    # Use coordinates directly from the selection group in the loop
    mobile_group = traj_u.select_atoms(traj_backbone_selection)

    for i, ts in enumerate(tqdm(traj_u.trajectory)):
        results["frame"].append(i)

        # Ensure mobile_group reflects current timestep coordinates
        current_coords = mobile_group.positions.copy() # Important: work with a copy for calculations

        # 1. RMSD to References (perform superposition for each calculation)
        rmsd_s = rms.rmsd(current_coords, start_ref_coords, superposition=True)
        rmsd_e = rms.rmsd(current_coords, end_ref_coords, superposition=True)
        results["rmsd_to_start"].append(rmsd_s)
        results["rmsd_to_end"].append(rmsd_e)

        # 2. Path Deviation Check
        is_deviating = (rmsd_s > deviation_rmsd_threshold_value and
                        rmsd_e > deviation_rmsd_threshold_value)
        results["is_deviating"].append(is_deviating)

        # 3. Frame-to-Frame RMSD (Path Smoothness)
        if prev_frame_coords is not None:
            # Superposition is important for frame-to-frame RMSD
            rmsd_f2f = rms.rmsd(current_coords, prev_frame_coords, superposition=True)
            results["rmsd_frame_to_frame"].append(rmsd_f2f)
            is_discontinuous = rmsd_f2f > smoothness_threshold  # 使用新计算的阈值
        else:
            results["rmsd_frame_to_frame"].append(0.0) # First frame has no previous frame
            is_discontinuous = False # Cannot be discontinuous
        results["is_discontinuous"].append(is_discontinuous)

        # Store current coordinates for the next iteration's frame-to-frame calculation
        prev_frame_coords = current_coords # This was already a copy

    # --- Analysis Summary ---
    print("\n--- Path Geometry Analysis Summary ---")
    results_np = {k: np.array(v) for k, v in results.items()} # Convert to numpy arrays

    max_f2f_rmsd = results_np['rmsd_frame_to_frame'][1:].max() if len(results['frame']) > 1 else 0.0
    print(f"Max Frame-to-Frame RMSD: {max_f2f_rmsd:.3f} Å")
    n_discontinuous = results_np['is_discontinuous'][1:].sum() # Skip first frame's False value
    print(f"Frames with potential discontinuity (Frame-to-Frame RMSD > {smoothness_threshold:.2f} Å): {n_discontinuous} ({n_discontinuous/max(1, len(results['frame'])-1)*100:.1f}%)")

    n_deviating = results_np['is_deviating'].sum()
    print(f"Frames marked as 'deviating' (RMSD to both ends > {deviation_rmsd_threshold_value:.2f} Å): {n_deviating} ({n_deviating/len(results['frame'])*100:.1f}%)")
    if n_deviating > 0:
        deviating_indices = results_np['frame'][results_np['is_deviating']]
        print(f"  Deviating frame indices: {deviating_indices[:10]}..." if len(deviating_indices) > 10 else deviating_indices)
    if n_discontinuous > 0:
        discontinuous_indices = results_np['frame'][results_np['is_discontinuous']]
        print(f"  Discontinuous frame indices: {discontinuous_indices[:10]}..." if len(discontinuous_indices) > 10 else discontinuous_indices)

    # --- Save Data ---
    data_filename = f"{output_prefix}_metrics.csv"
    try:
        header = ",".join(results.keys())
        # Need to handle boolean conversion for savetxt if needed or format appropriately
        data_out_list = []
        keys = list(results.keys())
        for i in range(len(results["frame"])):
             row = [results[k][i] for k in keys]
             data_out_list.append(row)
        data_out = np.array(data_out_list)

        # Format booleans as integers (0 or 1) for easier CSV handling
        bool_cols_indices = [keys.index(k) for k in ['is_deviating', 'is_discontinuous']]
        data_out[:, bool_cols_indices] = data_out[:, bool_cols_indices].astype(int)

        np.savetxt(data_filename, data_out, delimiter=",", header=header, fmt=['%d'] + ['%.4f'] * 3 + ['%d'] * 2, comments='') # Adjust format string based on columns
        print(f"Per-frame metrics saved to: {data_filename}")
    except Exception as e:
        print(f"Warning: Could not save metrics data to CSV. {e}")

    # --- Plotting ---
    if not args.skip_plots:
        print("Generating plots...")
        frames = results_np['frame']

        # Plot 1: Frame-to-Frame RMSD (Path Smoothness)
        fig_smooth, ax_smooth = plt.subplots(figsize=(10, 4))
        ax_smooth.plot(frames[1:], results_np['rmsd_frame_to_frame'][1:], label='RMSD (Å)', color='purple', linewidth=1)
        ax_smooth.axhline(smoothness_threshold, color='r', linestyle='--', label=f'Threshold ({smoothness_threshold:.3f} Å)')
        ax_smooth.set_xlabel('Frame Index')
        ax_smooth.set_ylabel('Frame-to-Frame RMSD (Å)')
        ax_smooth.set_title('Path Smoothness Analysis')
        ax_smooth.legend()
        ax_smooth.grid(True, alpha=0.5)
        fig_smooth.tight_layout()
        smooth_plot_filename = f"{output_prefix}_smoothness_plot.png"
        fig_smooth.savefig(smooth_plot_filename, dpi=150)
        print(f"Smoothness plot saved to: {smooth_plot_filename}")

        # Plot 2: Deviation RMSD Scatter Plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 7))
        # Color points by frame index to see progression (optional)
        scatter_colors = frames
        sc = ax_scatter.scatter(results_np['rmsd_to_start'], results_np['rmsd_to_end'],
                                c=scatter_colors, cmap='viridis', s=10, alpha=0.7,
                                label='Frames (color=Frame Index)')
        # Highlight deviating points
        deviating_mask = results_np['is_deviating']
        ax_scatter.scatter(results_np['rmsd_to_start'][deviating_mask], results_np['rmsd_to_end'][deviating_mask],
                           facecolors='none', edgecolors='red', s=30, label=f'Deviating (Both RMSD > {args.deviation_threshold_multiplier:.1f}*RMSDse = {deviation_rmsd_threshold_value:.2f} Å)')

        ax_scatter.set_xlabel(f"RMSD to Start ({os.path.basename(args.ref_start)}) (Å)")
        ax_scatter.set_ylabel(f"RMSD to End ({os.path.basename(args.ref_end)}) (Å)")
        ax_scatter.set_title('Path Deviation Analysis')
        ax_scatter.axhline(deviation_rmsd_threshold_value, color='grey', linestyle=':', alpha=0.7)
        ax_scatter.axvline(deviation_rmsd_threshold_value, color='grey', linestyle=':', alpha=0.7, label=f'Deviation Threshold ({deviation_rmsd_threshold_value:.2f}Å)')
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(fontsize='small')
        cbar = fig_scatter.colorbar(sc)
        cbar.set_label('Frame Index')
        fig_scatter.tight_layout()
        scatter_plot_filename = f"{output_prefix}_deviation_scatter.png"
        fig_scatter.savefig(scatter_plot_filename, dpi=150)
        print(f"Deviation scatter plot saved to: {scatter_plot_filename}")

    print("\nPath geometry evaluation complete.")


if __name__ == "__main__":
    main()
