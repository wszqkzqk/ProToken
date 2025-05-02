#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings
import logging
import numpy as np
import mdtraj as md
import joblib # To load models
import matplotlib.pyplot as plt
from itertools import combinations # If using distances/angles

# --- Configure warnings ---
warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Assume calculate_features function is available (MUST BE IDENTICAL to training) ---
# Copy the exact 'calculate_features' function used in the script that
# generated the selected_indices and tica_model files.
# It must return features_data (N_frames, N_total_features) and needs_alignment flag.
# Make sure all imports needed by calculate_features are present.
# --- Start Placeholder for calculate_features ---
def calculate_features(traj, logger, feature_type='backbone_torsions'):
    """Calculates backbone torsion features (phi/psi sin/cos) using MDTraj."""
    if feature_type != 'backbone_torsions':
        raise ValueError("This script requires feature_type to be 'backbone_torsions' matching the trained model.")

    logger.debug(f"Calculating backbone torsion angles (phi, psi) for {traj.n_frames} frames...")
    phi_indices, phi_angles = md.compute_phi(traj, periodic=False)
    psi_indices, psi_angles = md.compute_psi(traj, periodic=False)

    if phi_angles.size == 0 and psi_angles.size == 0:
         msg = f"No phi or psi angles could be computed for a trajectory ({traj})."
         logger.error(msg)
         raise ValueError(msg)

    phi_cos = np.cos(phi_angles); phi_sin = np.sin(phi_angles)
    psi_cos = np.cos(psi_angles); psi_sin = np.sin(psi_angles)
    phi_cos = np.nan_to_num(phi_cos, nan=0.0); phi_sin = np.nan_to_num(phi_sin, nan=0.0)
    psi_cos = np.nan_to_num(psi_cos, nan=0.0); psi_sin = np.nan_to_num(psi_sin, nan=0.0)

    feature_list = []
    if phi_angles.size > 0: feature_list.extend([phi_cos, phi_sin])
    if psi_angles.size > 0: feature_list.extend([psi_cos, psi_sin])
    if not feature_list: raise ValueError("Feature list empty after processing angles.")

    features_data = np.hstack(feature_list)
    feature_dim = features_data.shape[1]
    logger.debug(f"Calculated backbone torsions. Feature dimension: {feature_dim}")

    if features_data.size == 0 or feature_dim == 0:
        raise ValueError("Backbone torsion calculation resulted in empty/zero-dim feature array.")

    # Return float64 for consistency, needs_alignment is always False for torsions
    return features_data.astype(np.float64), False
# --- End Placeholder for calculate_features ---


# --- Setup Logging ---
def setup_logging(is_verbose=False):
    level = logging.DEBUG if is_verbose else logging.INFO
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
    return root_logger

# --- Function to process a single PDB (load, features, select, project) ---
def load_select_project(pdb_file, selected_indices, tica_model, logger):
    """Loads PDB, calculates all features, selects subset, projects onto TICA."""
    logger.info(f"Processing: {os.path.basename(pdb_file)}")
    try:
        # 1. Load Trajectory (or structure)
        traj = md.load(pdb_file)
        if traj.n_frames == 0: raise ValueError("File contains 0 frames.")
        logger.debug(f"Loaded {pdb_file} ({traj.n_frames} frames).")

        # 2. Calculate ALL raw features
        # Feature type is implicitly 'backbone_torsions' based on script design
        features_all, _ = calculate_features(traj, logger) # Needs_alignment is False for torsions
        logger.debug(f"Calculated all features shape: {features_all.shape}")

        # 3. Select the *specific* features used for TICA training
        if features_all.shape[1] != np.max(selected_indices) + 1 and features_all.shape[1] < np.max(selected_indices):
             # Basic check if indices are valid for the calculated features dimension
             msg = (f"Mismatch between number of calculated features ({features_all.shape[1]}) "
                    f"and the maximum selected index ({np.max(selected_indices)}). "
                    f"Ensure feature calculation is identical to training.")
             logger.error(msg)
             raise ValueError(msg)
        try:
            features_selected = features_all[:, selected_indices]
            logger.debug(f"Selected features shape: {features_selected.shape}")
        except IndexError as e:
            logger.error(f"Error selecting features using indices for {pdb_file}. "
                         f"Max index: {np.max(selected_indices)}, Feature shape: {features_all.shape}. Error: {e}")
            raise ValueError("Feature selection failed due to index mismatch.")

        # Dimension check against TICA model's expected input
        # Note: Deeptime models might not have a simple attribute like sklearn's n_features_in_
        # We rely on the selected_indices having been generated correctly for this model.
        # Add checks here if tica_model exposes expected input dimension.
        expected_dim = len(selected_indices)
        if features_selected.shape[1] != expected_dim:
             logger.warning(f"Selected feature dimension ({features_selected.shape[1]}) doesn't match "
                            f"expected based on indices ({expected_dim}). Check selection logic.")
             # Proceeding, but this could indicate an issue.

        # 4. Apply TICA Transformation
        logger.debug(f"Applying TICA transformation...")
        tica_output = tica_model.transform(features_selected)
        logger.info(f"Projected {os.path.basename(pdb_file)}. Output shape: {tica_output.shape}")
        return tica_output

    except Exception as e:
        logger.error(f"Failed to process {pdb_file}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None # Return None on failure

def main():
    parser = argparse.ArgumentParser(
        description='Visualize SMD and ProToken trajectories projected onto a pre-computed '
                    'difference-selected backbone torsion TICA model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input Files
    parser.add_argument('smd_pdb_file', help='Path to the multi-frame SMD trajectory PDB.')
    parser.add_argument('protoken_pdb_file', help='Path to the multi-frame ProToken trajectory PDB.')
    parser.add_argument('start_pdb_file', help='Path to the single-frame start reference PDB.')
    parser.add_argument('end_pdb_file', help='Path to the single-frame end reference PDB.')
    parser.add_argument('tica_model_file', help='Path to the saved TICA model file (.joblib) trained on difference-selected features.')
    parser.add_argument('selected_indices_file', help='Path to the .npy file containing the indices of the features used for TICA.')
    # Output Control
    parser.add_argument('-o', '--output-prefix', default="projection_comparison", help='Prefix for the output plot file (.png).')
    parser.add_argument('--output-dir', default=".", help='Directory to save the output plot.')
    parser.add_argument('--v', '--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()
    logger = setup_logging(args.v)

    # --- Input Validation ---
    required_files = [args.smd_pdb_file, args.protoken_pdb_file, args.tica_model_file,
                      args.selected_indices_file, args.start_pdb_file, args.end_pdb_file]
    for f in required_files:
        if not os.path.isfile(f):
            sys.exit(f"Error: Input file not found: {f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_plot_file = os.path.join(args.output_dir, f"{args.output_prefix}_plot.png")

    try:
        # 1. Load TICA Model and Selected Indices
        logger.info(f"Loading TICA model from: {args.tica_model_file}")
        tica_model = joblib.load(args.tica_model_file)
        logger.info(f"Loading selected feature indices from: {args.selected_indices_file}")
        selected_indices = np.load(args.selected_indices_file)
        logger.info(f"Loaded {len(selected_indices)} selected feature indices.")
        if selected_indices.ndim != 1:
            raise ValueError("Selected indices file should contain a 1D array.")

        # --- Process all PDBs using the loaded model and indices ---
        # Note: Alignment is not needed for backbone_torsions feature type.

        # Process SMD Trajectory
        tica_output_smd = load_select_project(args.smd_pdb_file, selected_indices, tica_model, logger)
        if tica_output_smd is None: sys.exit("Exiting due to error processing SMD trajectory.")

        # Process ProToken Trajectory
        tica_output_protoken = load_select_project(args.protoken_pdb_file, selected_indices, tica_model, logger)
        if tica_output_protoken is None: sys.exit("Exiting due to error processing ProToken trajectory.")

        # Process Start Reference
        tica_output_start = load_select_project(args.start_pdb_file, selected_indices, tica_model, logger)
        start_point = tica_output_start[0] if tica_output_start is not None else None

        # Process End Reference
        tica_output_end = load_select_project(args.end_pdb_file, selected_indices, tica_model, logger)
        end_point = tica_output_end[0] if tica_output_end is not None else None


        # --- Plotting ---
        logger.info(f"Generating comparison plot: {output_plot_file}")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine plot limits based on all trajectory data
        all_x = np.concatenate((tica_output_smd[:, 0], tica_output_protoken[:, 0]))
        all_y = np.concatenate((tica_output_smd[:, 1], tica_output_protoken[:, 1]))
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05

        # Plot SMD Trajectory
        smd_label = f'SMD: {os.path.basename(args.smd_pdb_file)}'
        scatter_smd = ax.scatter(
            tica_output_smd[:, 0], tica_output_smd[:, 1], label=smd_label,
            c=np.arange(len(tica_output_smd)), cmap='viridis',
            s=15, alpha=0.6, zorder=1
        )
        cbar_smd = fig.colorbar(scatter_smd, ax=ax, location='left', shrink=0.8, pad=0.08)
        cbar_smd.set_label(f'Frame Index ({os.path.basename(args.smd_pdb_file)})')

        # Plot ProToken Trajectory
        protoken_label = f'ProToken: {os.path.basename(args.protoken_pdb_file)}'
        scatter_pro = ax.scatter(
            tica_output_protoken[:, 0], tica_output_protoken[:, 1], label=protoken_label,
            c=np.arange(len(tica_output_protoken)), cmap='cool',
            s=15, alpha=0.7, zorder=2
        )
        cbar_pro = fig.colorbar(scatter_pro, ax=ax, location='right', shrink=0.8, pad=0.01)
        cbar_pro.set_label(f'Frame Index ({os.path.basename(args.protoken_pdb_file)})')

        # Plot Start/End Points
        if start_point is not None:
            ax.scatter(start_point[0], start_point[1], marker='^', s=180, c='red', edgecolors='black', zorder=5, label='Start Ref')
            logger.info(f"Start reference projected to: ({start_point[0]:.3f}, {start_point[1]:.3f})")
        if end_point is not None:
            ax.scatter(end_point[0], end_point[1], marker='s', s=180, c='blue', edgecolors='black', zorder=5, label='End Ref')
            logger.info(f"End reference projected to: ({end_point[0]:.3f}, {end_point[1]:.3f})")

        ax.set_xlabel("TICA Component 1 (IC1)")
        ax.set_ylabel("TICA Component 2 (IC2)")
        ax.set_title("TICA Projection Comparison (using Diff-Selected Torsions)")
        # Adjust limits slightly beyond data range
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2) # Adjust legend position
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0.05, 0, 0.95, 1]) # Adjust layout for colorbars
        plt.savefig(output_plot_file, dpi=300)
        logger.info(f"Comparison plot saved to: {output_plot_file}")

    except Exception as e:
        logger.error(f"An critical error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
