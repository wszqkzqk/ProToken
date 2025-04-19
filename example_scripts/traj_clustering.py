#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler # Optional but sometimes helpful for DBSCAN
import argparse
import os
import warnings
from tqdm import tqdm
import pandas as pd

# Suppress PDB reading warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.topology.PDBParser')

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory reliability using unsupervised clustering (DBSCAN) on path deviation and smoothness analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("trajectory", help="Path to the multi-frame PDB trajectory file.")
    parser.add_argument("ref_start", help="Path to the reference Start PDB file.")
    parser.add_argument("ref_end", help="Path to the reference End PDB file.")
    parser.add_argument(
        "--select",
        default="protein and (name CA)",
        help="MDAnalysis selection string for backbone atoms."
    )
    # --- DBSCAN Parameters ---
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=1.0, # Adjust based on typical point density in deviation plot
        help="DBSCAN eps parameter (maximum distance between samples for one to be considered as in the neighborhood of the other)."
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5,   # Adjust based on trajectory length and expected noise
        help="DBSCAN min_samples parameter (number of samples in a neighborhood for a point to be considered as a core point)."
    )
    parser.add_argument(
        "--scale_features",
        action="store_true",
        help="Scale RMSD_to_Start and RMSD_to_End features before DBSCAN (sometimes helps if scales differ vastly)."
         )
    # --- Smoothness Analysis Parameters ---
    parser.add_argument(
        "--smoothness_large_jump_ratio",
        type=float,
        default=5.0, # Jumps > 5x median jump might be interesting
        help="Ratio to median frame-to-frame RMSD to count as a 'large jump'."
         )
    # --- Output ---
    parser.add_argument("--output_prefix", default=None, help="Prefix for output plot files and data. Default is 'traj_eval_cluster' in the trajectory directory.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip generating plots.")

    args = parser.parse_args()
    
    if args.output_prefix is None:
        traj_dir = os.path.dirname(os.path.abspath(args.trajectory))
        args.output_prefix = os.path.join(traj_dir, "traj_eval_cluster")
    
    # --- Load Structures ---
    print("Loading structures...")
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
        # Ensure selection includes atoms needed for RMSD
        if not ("name N" in traj_backbone_selection and "name CA" in traj_backbone_selection and "name C" in traj_backbone_selection):
             print("Warning: Selection string might not contain all backbone atoms (N, CA, C). Ensure it's appropriate.")

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

    # --- Calculate Reference RMSDs (for scale mainly) ---
    start_ref_coords = start_backbone.positions.copy()
    end_ref_coords = end_backbone.positions.copy()
    rms.rmsd(end_ref_coords, start_ref_coords, superposition=True) # Align end_ref_coords in place
    rmsd_start_end = rms.rmsd(start_ref_coords, end_ref_coords, superposition=False)
    print(f"RMSD between aligned Start and End structures: {rmsd_start_end:.3f} Å")

    # --- Process Trajectory ---
    print(f"Processing {len(traj_u.trajectory)} frames to calculate RMSDs...")
    rmsd_to_start_list = []
    rmsd_to_end_list = []
    rmsd_f2f_list = []
    frames_indices = list(range(len(traj_u.trajectory)))

    prev_frame_coords = None
    mobile_group = traj_u.select_atoms(traj_backbone_selection) # Select once

    for i, ts in enumerate(tqdm(traj_u.trajectory)):
        current_coords = mobile_group.positions.copy()

        # RMSD to References
        rmsd_s = rms.rmsd(current_coords, start_ref_coords, superposition=True)
        rmsd_e = rms.rmsd(current_coords, end_ref_coords, superposition=True)
        rmsd_to_start_list.append(rmsd_s)
        rmsd_to_end_list.append(rmsd_e)

        # Frame-to-Frame RMSD
        if prev_frame_coords is not None:
            rmsd_f2f = rms.rmsd(current_coords, prev_frame_coords, superposition=True)
            rmsd_f2f_list.append(rmsd_f2f)
        else:
            # The list will be one element shorter than others, handle this later
            pass # No f2f RMSD for the first frame

        prev_frame_coords = current_coords

    # Combine results - ensure lists have compatible lengths
    rmsd_f2f_array = np.array([0.0] + rmsd_f2f_list) # Prepend 0 for first frame
    deviation_data = np.array([rmsd_to_start_list, rmsd_to_end_list]).T # Shape (n_frames, 2)

    # --- Path Deviation Clustering (DBSCAN) ---
    print(f"\nPerforming DBSCAN clustering (eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples})...")

    X_deviation = deviation_data
    if args.scale_features:
        print("Scaling features before clustering.")
        scaler = StandardScaler()
        X_deviation = scaler.fit_transform(deviation_data)

    db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit(X_deviation)
    labels = db.labels_ # Cluster labels for each point. Noise is labeled -1.

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    noise_percentage = (n_noise / len(labels)) * 100 if len(labels) > 0 else 0

    print(f"DBSCAN found {n_clusters} cluster(s).")
    print(f"{n_noise} points classified as noise ({noise_percentage:.1f}%).")

    # --- Path Smoothness Analysis ---
    print("\nAnalyzing path smoothness...")
    rmsd_f2f_valid = rmsd_f2f_array[1:] # Exclude the initial 0.0
    max_f2f_rmsd = np.max(rmsd_f2f_valid) if len(rmsd_f2f_valid) > 0 else 0.0
    mean_f2f_rmsd = np.mean(rmsd_f2f_valid) if len(rmsd_f2f_valid) > 0 else 0.0
    median_f2f_rmsd = np.median(rmsd_f2f_valid) if len(rmsd_f2f_valid) > 0 else 0.0
    std_f2f_rmsd = np.std(rmsd_f2f_valid) if len(rmsd_f2f_valid) > 0 else 0.0
    percentile_95_f2f_rmsd = np.percentile(rmsd_f2f_valid, 95) if len(rmsd_f2f_valid) > 0 else 0.0

    large_jump_threshold = args.smoothness_large_jump_ratio * median_f2f_rmsd
    n_large_jumps = np.sum(rmsd_f2f_valid > large_jump_threshold) if median_f2f_rmsd > 1e-6 and len(rmsd_f2f_valid) > 0 else 0 # Avoid division by zero or thresholding noise

    print(f"Max Frame-to-Frame RMSD: {max_f2f_rmsd:.3f} Å")
    print(f"Mean Frame-to-Frame RMSD: {mean_f2f_rmsd:.3f} Å")
    print(f"Median Frame-to-Frame RMSD: {median_f2f_rmsd:.3f} Å")
    print(f"Std Dev Frame-to-Frame RMSD: {std_f2f_rmsd:.3f} Å")
    print(f"95th Percentile Frame-to-Frame RMSD: {percentile_95_f2f_rmsd:.3f} Å")
    print(f"Number of large jumps (> {args.smoothness_large_jump_ratio:.1f} * median = {large_jump_threshold:.3f} Å): {n_large_jumps}")


    # --- Save Combined Data ---
    output_data = pd.DataFrame({
        'frame': frames_indices,
        'rmsd_to_start': rmsd_to_start_list,
        'rmsd_to_end': rmsd_to_end_list,
        'rmsd_f2f': rmsd_f2f_array, # Includes the prepended 0
        'dbscan_cluster': labels
    })
    data_filename = f"{args.output_prefix}_metrics_clusters.csv"
    try:
        output_data.to_csv(data_filename, index=False, float_format='%.4f')
        print(f"\nPer-frame metrics and clusters saved to: {data_filename}")
    except Exception as e:
        print(f"Warning: Could not save metrics data to CSV. {e}")

    # --- Interpretation Guidance ---
    print("\n--- Interpretation Guidance ---")
    print("Consider a trajectory potentially LESS reliable if:")
    if n_clusters > 1:
        print(f"  - Path Deviation plot shows MULTIPLE clusters ({n_clusters} found). This may indicate jumps or discontinuities.")
    else:
        print("  - Path Deviation plot shows roughly ONE cluster (potentially good sign).")
    if noise_percentage > 10.0: # Example threshold for interpretation
        print(f"  - Path Deviation plot has a HIGH percentage of noise points ({noise_percentage:.1f}%). This may indicate many intermediate outlier structures.")
    if max_f2f_rmsd > 3.0: # Example threshold for interpretation
        print(f"  - Path Smoothness shows very LARGE maximum jump ({max_f2f_rmsd:.2f} Å).")
    if n_large_jumps > 5: # Example threshold for interpretation
         print(f"  - Path Smoothness shows MANY large jumps ({n_large_jumps} found > {large_jump_threshold:.2f} Å).")
    print("Consider a trajectory potentially MORE reliable if:")
    print("  - Path Deviation plot shows ONE dominant cluster (n_clusters=1).")
    print(f"  - Path Deviation plot has LOW noise percentage (<5-10%).")
    print(f"  - Path Smoothness shows LOW maximum jump (<1.5-2.0 Å) and FEW/NO large relative jumps.")
    print("NOTE: These are guidelines; scientific judgment is crucial. Examine the plots.")


    # --- Plotting ---
    if not args.skip_plots:
        print("\nGenerating plots...")

        # Plot 1: Path Deviation Scatter Plot, colored by DBSCAN Cluster
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 7))
        unique_labels = set(labels)
        n_clusters_real = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Use distinctive color mapping to ensure clear differentiation between clusters
        if n_clusters_real > 10:
            # For many clusters, use continuous colormap
            colormap = plt.cm.jet
        else:
            # For fewer clusters, use discrete colormap to enhance distinction
            colormap = plt.cm.tab10
            
        colors = colormap(np.linspace(0, 1, max(n_clusters_real, 1)))
        color_idx = 0
        
        for k in sorted(unique_labels):
            if k == -1:
                # Use black for noise points
                col = [0, 0, 0, 1]
                marker = 'x'
                markersize = 6
                label = 'Noise / Outliers'
            else:
                col = colors[color_idx]
                color_idx += 1
                marker = 'o'
                markersize = 5
                label = f'Cluster {k}'

            class_member_mask = (labels == k)
            xy = deviation_data[class_member_mask]
            ax_scatter.scatter(xy[:, 0], xy[:, 1], 
                           marker=marker, 
                           s=markersize**2.5, 
                           c=[col], 
                           edgecolors='none',  # Remove borders for better visibility in dense regions
                           alpha=0.8 if k != -1 else 0.6, 
                           label=label)

        # Add reference positions for start and end points
        min_x, max_x = ax_scatter.get_xlim()
        min_y, max_y = ax_scatter.get_ylim()

        ax_scatter.set_xlabel(f"RMSD to Start ({os.path.basename(args.ref_start)}) (Å)")
        ax_scatter.set_ylabel(f"RMSD to End ({os.path.basename(args.ref_end)}) (Å)")
        ax_scatter.set_title(f'Path Deviation Analysis (DBSCAN: {n_clusters} clusters, {noise_percentage:.1f}% noise)')
        ax_scatter.legend(fontsize='small', loc='best', framealpha=0.7)
        ax_scatter.grid(True, alpha=0.3)
        
        # Set axis ranges to better display the data
        x_range = max_x - min_x
        y_range = max_y - min_y
        ax_scatter.set_xlim(min_x - 0.05 * x_range, max_x + 0.05 * x_range)
        ax_scatter.set_ylim(min_y - 0.05 * y_range, max_y + 0.05 * y_range)
        
        fig_scatter.tight_layout()
        scatter_plot_filename = f"{args.output_prefix}_deviation_clustered.png"
        fig_scatter.savefig(scatter_plot_filename, dpi=150)
        print(f"Clustered deviation scatter plot saved to: {scatter_plot_filename}")


        # Plot 2: Path Smoothness Plot
        fig_smooth, ax_smooth = plt.subplots(figsize=(12, 4)) # Wider plot
        ax_smooth.plot(frames_indices[1:], rmsd_f2f_list, label='RMSD (Å)', color='purple', linewidth=1)
        ax_smooth.axhline(median_f2f_rmsd, color='grey', linestyle=':', alpha=0.8, label=f'Median ({median_f2f_rmsd:.2f} Å)')
        ax_smooth.axhline(percentile_95_f2f_rmsd, color='orange', linestyle='--', alpha=0.8, label=f'95th Percentile ({percentile_95_f2f_rmsd:.2f} Å)')
        if median_f2f_rmsd > 1e-6: # Add line for large jump threshold if median is not zero
             ax_smooth.axhline(large_jump_threshold, color='red', linestyle='--', alpha=0.8, label=f'Large Jump Thr. ({large_jump_threshold:.2f} Å)')

        ax_smooth.set_xlabel('Frame Index')
        ax_smooth.set_ylabel('Frame-to-Frame RMSD (Å)')
        ax_smooth.set_title('Path Smoothness Analysis')
        ax_smooth.legend(fontsize='small')
        ax_smooth.grid(True, alpha=0.5)
        # Optionally set y-limit if extreme values distort the plot too much
        # upper_ylim = max(5.0, percentile_95_f2f_rmsd * 1.5) # Example adaptive limit
        # ax_smooth.set_ylim(bottom=-0.1, top=upper_ylim)
        fig_smooth.tight_layout()
        smooth_plot_filename = f"{args.output_prefix}_smoothness_plot.png"
        fig_smooth.savefig(smooth_plot_filename, dpi=150)
        print(f"Smoothness plot saved to: {smooth_plot_filename}")

        # plt.show() # Optionally show plots interactively

    print("\nUnsupervised evaluation complete.")

if __name__ == "__main__":
    main()
