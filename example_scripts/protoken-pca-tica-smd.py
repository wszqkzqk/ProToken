#!/usr/bin/env python3

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

def calculate_features(traj, feature_type, logger):
    topology = traj.topology
    features_data = None
    needs_alignment = False
    feature_dim = 0
    if feature_type == 'ca_dist':
        ca_indices = topology.select('name CA and protein')
        if len(ca_indices) < 2: raise ValueError("Not enough CAs")
        ca_pairs = list(combinations(ca_indices, 2))
        if not ca_pairs: raise ValueError("No CA pairs")
        features_data = md.compute_distances(traj, ca_pairs, periodic=False)
        feature_dim = features_data.shape[1]
        logger.debug(f"Calculated C-alpha distances. Dim: {feature_dim}") # Changed to debug
    elif feature_type == 'heavy_atom':
        heavy_atom_indices = topology.select('not element H and protein')
        if len(heavy_atom_indices) == 0: raise ValueError("No heavy atoms")
        features_data = traj.xyz[:, heavy_atom_indices, :].reshape(traj.n_frames, -1)
        feature_dim = features_data.shape[1]
        needs_alignment = True
        logger.debug(f"Extracted heavy atom coordinates. Dim: {feature_dim}") # Changed to debug
    elif feature_type == 'backbone_torsions':
        phi_indices, phi_angles = md.compute_phi(traj, periodic=False)
        psi_indices, psi_angles = md.compute_psi(traj, periodic=False)
        phi_cos = np.cos(phi_angles); phi_sin = np.sin(phi_angles)
        psi_cos = np.cos(psi_angles); psi_sin = np.sin(psi_angles)
        phi_cos = np.nan_to_num(phi_cos, nan=0.0); phi_sin = np.nan_to_num(phi_sin, nan=0.0)
        psi_cos = np.nan_to_num(psi_cos, nan=0.0); psi_sin = np.nan_to_num(psi_sin, nan=0.0)
        features_data = np.hstack([phi_cos, phi_sin, psi_cos, psi_sin])
        if features_data.size == 0: raise ValueError("Empty torsion features")
        feature_dim = features_data.shape[1]
        logger.debug(f"Calculated backbone torsions (phi/psi cos/sin). Dim: {feature_dim}") # Changed to debug
    else:
        raise ValueError(f"Invalid feature type '{feature_type}'")
    if features_data is None or features_data.shape[0] == 0 or features_data.shape[1] == 0 :
        raise ValueError(f"Feature calculation failed for '{feature_type}'")
    return features_data.astype(np.float64), needs_alignment

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

# --- Function to process a single trajectory (load, align, features, project) ---
def process_trajectory(pdb_file, feature_type, pca_model, tica_model, ref_for_align, logger):
    """Loads, optionally aligns, computes features, and projects a trajectory."""
    logger.info(f"Processing trajectory: {pdb_file}")

     # 1. Load Trajectory
    try:
        traj = md.load(pdb_file)
        if traj.n_frames == 0: raise ValueError("Trajectory has 0 frames.")
        logger.info(f"Loaded trajectory with {traj.n_frames} frames.")
    except Exception as e:
        logger.error(f"Failed to load {pdb_file}: {e}")
        return None # Return None on failure

    # 2. Check if alignment is needed based on feature type
    try:
        # Use first frame to determine alignment need without loading full features yet
        tmp_feat, needs_alignment_flag = calculate_features(traj[0], feature_type, logger)
    except Exception as e:
        logger.error(f"Could not determine alignment need for {pdb_file} (feature calculation on first frame failed): {e}")
        return None

    # 3. Align Trajectory (if required) using the provided reference
    ref_structure = None
    if needs_alignment_flag:
        if ref_for_align:
            try:
                logger.info(f"Aligning {pdb_file} to reference...")
                heavy_atom_indices = traj.topology.select('not element H and protein')
                if len(heavy_atom_indices) == 0:
                     logger.warning("No heavy atoms found for alignment in {pdb_file}. Skipping alignment.")
                else:
                     # Perform alignment inplace
                     traj.superpose(ref_for_align, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)
                     logger.info("Alignment complete.")
            except Exception as e:
                 logger.warning(f"Alignment failed for {pdb_file}: {e}. Using unaligned coordinates.")
        else:
            logger.warning(f"Feature type {feature_type} requires alignment, but no reference was provided for {pdb_file}. Using unaligned coordinates.")

    # 4. Calculate Features
    try:
        logger.info(f"Calculating '{feature_type}' features for {pdb_file}...")
        features_data, _ = calculate_features(traj, feature_type, logger)
        logger.info(f"Features shape: {features_data.shape}")

        # Dimension Check (against PCA model)
        if hasattr(pca_model, 'n_features_in_') and features_data.shape[1] != pca_model.n_features_in_:
             msg = f"Feature dimension mismatch for {pdb_file}! Expected {pca_model.n_features_in_}, got {features_data.shape[1]}."
             logger.error(msg)
             raise ValueError(msg)

    except Exception as e:
        logger.error(f"Feature calculation failed for {pdb_file}: {e}")
        return None

    # 5. Apply PCA Transformation
    try:
        logger.info(f"Applying PCA transformation to {pdb_file} features...")
        pca_output = pca_model.transform(features_data)
        logger.info(f"PCA output shape: {pca_output.shape}")
    except Exception as e:
        logger.error(f"PCA transformation failed for {pdb_file}: {e}")
        return None

    # 6. Apply TICA Transformation
    try:
        logger.info(f"Applying TICA transformation to {pdb_file} PCA output...")
        tica_output = tica_model.transform(pca_output)
        logger.info(f"Final TICA projection shape: {tica_output.shape}")
    except Exception as e:
         logger.error(f"TICA transformation failed for {pdb_file}: {e}")
         return None

    return tica_output


def main():
    parser = argparse.ArgumentParser(
        description='Project original training and a new PDB trajectory onto a pre-trained PCA+TICA model and plot them together.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input Trajectories
    parser.add_argument(
        'orig_pdb_file',
        help='Path to the multi-frame PDB file used for TRAINING the PCA/TICA models.'
    )
    parser.add_argument(
        'new_pdb_file',
        help='Path to the new multi-frame PDB file to project and compare.'
    )
    # Models
    parser.add_argument(
        'pca_model_file',
        help='Path to the saved PCA model file (.joblib).'
    )
    parser.add_argument(
        'tica_model_file',
        help='Path to the saved TICA model file (.joblib).'
    )
    # Feature and Alignment
    parser.add_argument(
        '--feature-type', default='backbone_torsions', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'],
        help='Feature type used to train the models (must match).'
    )
    # References (Optional Start/End, Required for Alignment)
    parser.add_argument(
        '--ref-pdb-for-align', default=None,
        help='Path to the single frame PDB used as reference for alignment during training (REQUIRED if feature_type is heavy_atom). This file is used to load the reference structure.'
    )
    parser.add_argument(
        '--start-pdb', default=None,
        help='Optional path to a PDB file representing the starting structure (single frame).'
    )
    parser.add_argument(
        '--end-pdb', default=None,
        help='Optional path to a PDB file representing the ending structure (single frame).'
    )
    # Output Control
    parser.add_argument(
        '-o', '--output-prefix', default="compare_proj",
        help='Prefix for output files (projection data .npy and plot .png).'
    )
    parser.add_argument(
        '--output-dir', default="compare_traj",
        help='Directory to save the output files.'
    )
    parser.add_argument(
        '--v', '--verbose', action='store_true', help='Enable verbose logging.'
    )

    args = parser.parse_args()
    logger = setup_logging(args.v)

    # --- Input Validation ---
    if not os.path.isfile(args.orig_pdb_file): sys.exit(f"Error: Original PDB not found: {args.orig_pdb_file}")
    if not os.path.isfile(args.new_pdb_file): sys.exit(f"Error: New PDB not found: {args.new_pdb_file}")
    if not os.path.isfile(args.pca_model_file): sys.exit(f"Error: PCA model not found: {args.pca_model_file}")
    if not os.path.isfile(args.tica_model_file): sys.exit(f"Error: TICA model not found: {args.tica_model_file}")
    if args.feature_type == 'heavy_atom' and args.ref_pdb_for_align is None:
         sys.exit("Error: --ref-pdb-for-align is required for feature_type 'heavy_atom'.")
    if args.ref_pdb_for_align is not None and not os.path.isfile(args.ref_pdb_for_align):
         sys.exit(f"Error: Alignment reference PDB not found: {args.ref_pdb_for_align}")
    if args.start_pdb is not None and not os.path.isfile(args.start_pdb):
         logger.warning(f"Start PDB not found: {args.start_pdb}. Skipping start point.")
         args.start_pdb = None
    if args.end_pdb is not None and not os.path.isfile(args.end_pdb):
         logger.warning(f"End PDB not found: {args.end_pdb}. Skipping end point.")
         args.end_pdb = None

    os.makedirs(args.output_dir, exist_ok=True)
    output_projection_orig_file = os.path.join(args.output_dir, f"{args.output_prefix}_orig_tica_projection.npy")
    output_projection_new_file = os.path.join(args.output_dir, f"{args.output_prefix}_new_tica_projection.npy")
    output_plot_file = os.path.join(args.output_dir, f"{args.output_prefix}_comparison_plot.png")

    try:
        # 1. Load Models
        logger.info(f"Loading PCA model from: {args.pca_model_file}")
        pca_model = joblib.load(args.pca_model_file)
        logger.info(f"Loading TICA model from: {args.tica_model_file}")
        tica_model = joblib.load(args.tica_model_file)

        # 2. Determine Alignment Reference Structure (if needed)
        ref_align_struct = None
        if args.feature_type == 'heavy_atom' and args.ref_pdb_for_align:
            try:
                logger.info(f"Loading reference structure for alignment from {args.ref_pdb_for_align}")
                ref_align_struct = md.load(args.ref_pdb_for_align)[0] # Use first frame
            except Exception as e:
                 logger.error(f"Failed to load alignment reference {args.ref_pdb_for_align}: {e}. Aborting alignment.")
                 sys.exit(1)

        # 3. Process Original Trajectory
        tica_output_orig = process_trajectory(
            args.orig_pdb_file, args.feature_type, pca_model, tica_model, ref_align_struct, logger
        )
        if tica_output_orig is None:
            sys.exit(f"Failed to process original trajectory: {args.orig_pdb_file}")
        logger.info(f"Saving original trajectory projection to: {output_projection_orig_file}")
        np.save(output_projection_orig_file, tica_output_orig)

        # 4. Process New Trajectory
        tica_output_new = process_trajectory(
            args.new_pdb_file, args.feature_type, pca_model, tica_model, ref_align_struct, logger
        )
        if tica_output_new is None:
             sys.exit(f"Failed to process new trajectory: {args.new_pdb_file}")
        logger.info(f"Saving new trajectory projection to: {output_projection_new_file}")
        np.save(output_projection_new_file, tica_output_new)

        # 5. Process Optional Start/End Structures
        tica_start_point = None
        if args.start_pdb:
             start_proj = process_trajectory(args.start_pdb, args.feature_type, pca_model, tica_model, ref_align_struct, logger)
             if start_proj is not None:
                 tica_start_point = start_proj[0] # Assume single frame, take first row

        tica_end_point = None
        if args.end_pdb:
             end_proj = process_trajectory(args.end_pdb, args.feature_type, pca_model, tica_model, ref_align_struct, logger)
             if end_proj is not None:
                 tica_end_point = end_proj[0] # Assume single frame

        # 6. Plotting
        logger.info(f"Generating comparison plot: {output_plot_file}")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot Original Trajectory
        scatter_orig = ax.scatter(
            tica_output_orig[:, 0], tica_output_orig[:, 1],
            label=f'Orig: {os.path.basename(args.orig_pdb_file)}',
            c=np.arange(len(tica_output_orig)),
            cmap='viridis',
            s=15, alpha=0.5, zorder=1 # Lower zorder to be behind new traj if overlapping
        )
        cbar_orig = fig.colorbar(scatter_orig, ax=ax, location='left', shrink=0.8)
        cbar_orig.set_label(f'Frame Index ({os.path.basename(args.orig_pdb_file)})')

        # Plot New Trajectory
        scatter_new = ax.scatter(
            tica_output_new[:, 0], tica_output_new[:, 1],
            label=f'New: {os.path.basename(args.new_pdb_file)}',
            c=np.arange(len(tica_output_new)),
            cmap='cool', # Different colormap
            s=15, alpha=0.7, zorder=2 # Higher zorder
        )
        cbar_new = fig.colorbar(scatter_new, ax=ax, location='right', shrink=0.8)
        cbar_new.set_label(f'Frame Index ({os.path.basename(args.new_pdb_file)})')

        # Plot Start/End Points
        if tica_start_point is not None:
            ax.scatter(tica_start_point[0], tica_start_point[1], marker='^', s=150, c='red', edgecolors='black', zorder=5, label='Start Ref')
            logger.info(f"Start reference projected to: {tica_start_point[:2]}") # Show first 2 coords
        if tica_end_point is not None:
            ax.scatter(tica_end_point[0], tica_end_point[1], marker='s', s=150, c='blue', edgecolors='black', zorder=5, label='End Ref')
            logger.info(f"End reference projected to: {tica_end_point[:2]}") # Show first 2 coords

        ax.set_xlabel("TICA Component 1 (IC1)")
        ax.set_ylabel("TICA Component 2 (IC2)")
        ax.set_title("PCA+TICA Projection Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(output_plot_file, dpi=300)
        logger.info(f"Comparison plot saved to: {output_plot_file}")

    except Exception as e:
        logger.error(f"An critical error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
