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

def calculate_rmsds_to_references(universe, ref_start_coords, ref_end_coords, selection):
    """Calculates RMSD to start and end references for each frame."""
    rmsds_start = []
    rmsds_end = []
    mobile_group = universe.select_atoms(selection)

    for ts in tqdm(universe.trajectory, desc=f"Calculating RMSDs for {os.path.basename(universe.filename)}", leave=False):
        current_coords = mobile_group.positions.copy()
        # Superposition is crucial for meaningful RMSD comparison
        rmsd_s = rms.rmsd(current_coords, ref_start_coords, superposition=True)
        rmsd_e = rms.rmsd(current_coords, ref_end_coords, superposition=True)
        rmsds_start.append(rmsd_s)
        rmsds_end.append(rmsd_e)
    return np.array(rmsds_start), np.array(rmsds_end)

def main():
    parser = argparse.ArgumentParser(
        description="Compare Steered MD and ProToken trajectories in the RMSD-to-references space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("ref_start", help="Path to the reference Start PDB file.")
    parser.add_argument("ref_end", help="Path to the reference End PDB file.")
    parser.add_argument("smd_traj", help="Path to the Steered MD multi-frame PDB trajectory file.")
    parser.add_argument("protoken_traj", help="Path to the ProToken multi-frame PDB trajectory file.")
    parser.add_argument(
        "--select",
        default="protein and (name CA)",
        help="MDAnalysis selection string for backbone atoms."
    )
    parser.add_argument("--smd_color", default='blue', help="Color for Steered MD trajectory points.")
    parser.add_argument("--protoken_color", default='chocolate', help="Color for ProToken trajectory points.")
    parser.add_argument("--smd_label", default='Steered MD', help="Label for Steered MD trajectory in legend.")
    parser.add_argument("--protoken_label", default='ProToken', help="Label for ProToken trajectory in legend.")
    parser.add_argument("--output_png",
                        default=None,
                        help="Output filename for the PNG plot.")
    parser.add_argument("--plot_title", default="Trajectory Comparison in RMSD space", help="Title for the plot.")

    args = parser.parse_args()

    smd_dir = os.path.dirname(args.smd_traj) or '.'
    args.output_png = args.output_png or os.path.join(smd_dir, "trajectory_comparison.png")

    # --- Load Structures ---
    print("Loading structures...")
    try:
        smd_u = mda.Universe(args.smd_traj)
        protoken_u = mda.Universe(args.protoken_traj)
        start_u = mda.Universe(args.ref_start)
        end_u = mda.Universe(args.ref_end)
    except Exception as e:
        print(f"Error loading PDB files: {e}")
        return

    # --- Select Atoms and Check Consistency ---
    print(f"Using selection: '{args.select}'")
    try:
        start_backbone = start_u.select_atoms(args.select)
        end_backbone = end_u.select_atoms(args.select)
        smd_backbone_check = smd_u.select_atoms(args.select)
        protoken_backbone_check = protoken_u.select_atoms(args.select)

        n_atoms_start = len(start_backbone)
        n_atoms_end = len(end_backbone)
        n_atoms_smd = len(smd_backbone_check)
        n_atoms_protoken = len(protoken_backbone_check)

        if not (n_atoms_start == n_atoms_end == n_atoms_smd == n_atoms_protoken):
            print(f"Error: Atom count mismatch in selection!")
            print(f"  Start: {n_atoms_start}, End: {n_atoms_end}")
            print(f"  SMD Traj: {n_atoms_smd}, ProToken Traj: {n_atoms_protoken}")
            return
        if n_atoms_start == 0:
            print(f"Error: Selection resulted in 0 atoms.")
            return
        print(f"Consistent atom count ({n_atoms_start}) found across all structures.")
    except Exception as e:
        print(f"Error during atom selection: {e}")
        return

    # --- Prepare Reference Coordinates ---
    # Align references once to get stable coordinates for comparison
    start_ref_coords = start_backbone.positions.copy()
    end_ref_coords = end_backbone.positions.copy()
    rms.rmsd(end_ref_coords, start_ref_coords, superposition=True) # Align end to start
    # Use these aligned coordinates for all subsequent RMSD calculations against refs

    # Calculate RMSD between the aligned references
    rmsd_start_vs_end_aligned = rms.rmsd(start_ref_coords, end_ref_coords, superposition=True)
    print(f"RMSD between aligned Start and End structures: {rmsd_start_vs_end_aligned:.3f} Å (used for reference lines)")

    # --- Calculate RMSDs for Trajectories ---
    print("\nCalculating RMSDs for Steered MD trajectory...")
    smd_rmsds_start, smd_rmsds_end = calculate_rmsds_to_references(smd_u, start_ref_coords, end_ref_coords, args.select)

    print("\nCalculating RMSDs for ProToken trajectory...")
    protoken_rmsds_start, protoken_rmsds_end = calculate_rmsds_to_references(protoken_u, start_ref_coords, end_ref_coords, args.select)

    # --- Plotting ---
    print("\nGenerating comparison plot...")
    #plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot for SMD
    ax.scatter(smd_rmsds_start, smd_rmsds_end,
               c=args.smd_color,
               label=args.smd_label,
               alpha=0.6, s=15, edgecolors='w', linewidth=0.5) # Add white edge for clarity

    # Scatter plot for ProToken
    ax.scatter(protoken_rmsds_start, protoken_rmsds_end,
               c=args.protoken_color,
               label=args.protoken_label,
               alpha=0.6, s=15, edgecolors='w', linewidth=0.5)

    ax.axhline(rmsd_start_vs_end_aligned, color='grey', linestyle=':', alpha=0.8,
               label=f'RMSD(Start, End) = {rmsd_start_vs_end_aligned:.2f} Å')
    ax.axvline(rmsd_start_vs_end_aligned, color='grey', linestyle=':', alpha=0.8)


    # Add labels, title, legend
    ax.set_xlabel(f"RMSD to Start (Å)", fontsize=12)
    ax.set_ylabel(f"RMSD to End (Å)", fontsize=12)
    ax.set_title(args.plot_title, fontsize=14)

    # Ensure plot limits cover the data range and reference lines
    all_x = np.concatenate((smd_rmsds_start, protoken_rmsds_start))
    all_y = np.concatenate((smd_rmsds_end, protoken_rmsds_end))
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    buffer = 0.5 # Add some buffer
    ax.set_xlim(max(-buffer, x_min - buffer), max(x_max + buffer, rmsd_start_vs_end_aligned + buffer))
    ax.set_ylim(max(-buffer, y_min - buffer), max(y_max + buffer, rmsd_start_vs_end_aligned + buffer))


    legend = ax.legend(loc='best', fontsize='medium')
    #legend.get_frame().set_edgecolor('black')


    ax.set_aspect('equal', adjustable='box') # Make axes visually comparable
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7) # Customize grid

    # Save the plot
    plt.tight_layout()
    try:
        plt.savefig(args.output_png, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {args.output_png}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # plt.show() # Uncomment to display the plot interactively


if __name__ == "__main__":
    main()