#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings
import logging
from itertools import combinations

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import mdtraj as md
    import deeptime
    from deeptime.decomposition import TICA
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib # Using joblib for saving model objects
    # PCA is NOT imported/used in this version
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"Error: Missing required library. Please install deeptime, MDTraj, Matplotlib, NumPy, joblib, argparse.")
    logging.error(f"Run: pip install deeptime mdtraj matplotlib numpy joblib argparse")
    logging.error(f"Import error details: {e}")
    sys.exit(1)

# --- setup_logging and calculate_features remain the same ---
def setup_logging(log_file, is_verbose=False):
    """Configure logger."""
    level = logging.DEBUG if is_verbose else logging.INFO
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
    return root_logger

def calculate_features(traj, logger, feature_type='backbone_torsions'):
    """Calculates backbone torsion features (phi/psi sin/cos) using MDTraj."""
    if feature_type != 'backbone_torsions':
        raise ValueError("This script version currently only supports 'backbone_torsions' for difference-based selection.")

    logger.debug(f"Calculating backbone torsion angles (phi, psi) for {traj.n_frames} frames...")
    phi_indices, phi_angles = md.compute_phi(traj, periodic=False)
    psi_indices, psi_angles = md.compute_psi(traj, periodic=False)

    if phi_angles.size == 0 and psi_angles.size == 0:
         msg = "No phi or psi angles could be computed."
         logger.error(msg)
         raise ValueError(msg)

    # Use consistent order, e.g., cos(phi), sin(phi), cos(psi), sin(psi)
    phi_cos = np.cos(phi_angles)
    phi_sin = np.sin(phi_angles)
    psi_cos = np.cos(psi_angles)
    psi_sin = np.sin(psi_angles)

    # Handle potential NaNs from terminal residues etc.
    phi_cos = np.nan_to_num(phi_cos, nan=0.0)
    phi_sin = np.nan_to_num(phi_sin, nan=0.0)
    psi_cos = np.nan_to_num(psi_cos, nan=0.0)
    psi_sin = np.nan_to_num(psi_sin, nan=0.0)

    # Store features and indices (assuming standard phi/psi order from mdtraj)
    feature_list = []
    feature_desc = [] # To track which feature index corresponds to which angle pair

    n_phi = phi_angles.shape[1] if phi_angles.size > 0 else 0
    n_psi = psi_angles.shape[1] if psi_angles.size > 0 else 0

    angle_pair_index = 0
    if n_phi > 0:
         feature_list.extend([phi_cos, phi_sin])
         # feature_desc needs careful mapping if indices needed later; simplified for now
         for i in range(n_phi): feature_desc.append(f'phi_pair_{angle_pair_index+i}')
         angle_pair_index += n_phi
    if n_psi > 0:
         feature_list.extend([psi_cos, psi_sin])
         for i in range(n_psi): feature_desc.append(f'psi_pair_{angle_pair_index+i}')

    if not feature_list:
         msg = "Feature list is empty after processing angles."
         logger.error(msg)
         raise ValueError(msg)

    features_data = np.hstack(feature_list)
    feature_dim = features_data.shape[1]
    logger.debug(f"Calculated backbone torsions. Feature dimension: {feature_dim}")

    if features_data.size == 0 or feature_dim == 0:
        msg = "Backbone torsion calculation resulted in empty/zero-dim feature array."
        logger.error(msg)
        raise ValueError(msg)

    # Return as float64, descriptions might be useful later if needed
    return features_data.astype(np.float64), feature_desc


def run_diff_tica_analysis(traj_pdb_file, start_pdb_file, end_pdb_file,
                           diff_percentile, lag_time, n_dim,
                           output_dir, output_basename, plot_type):
    """
    Performs TICA analysis using features selected based on difference
    between start and end structures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define output filenames including percentile info
    percentile_tag = f"_diff{int(diff_percentile*100)}"
    log_filename = f"{output_basename}_backbone_torsions{percentile_tag}_tica_lag{lag_time}_dim{n_dim}.log"
    output_log_file = os.path.join(output_dir, log_filename)
    # Set is_verbose=False for setup_logging, control via handler level later if needed
    logger = setup_logging(output_log_file, is_verbose=False)

    logger.info("--- Starting Difference-Selected TICA Analysis ---")
    logger.info(f"Deeptime version: {deeptime.__version__}")
    logger.info(f"MDTraj version: {md.__version__}")
    logger.info(f"Trajectory PDB: {traj_pdb_file}")
    logger.info(f"Start PDB: {start_pdb_file}")
    logger.info(f"End PDB: {end_pdb_file}")
    logger.info(f"Feature type: backbone_torsions (phi/psi, sin/cos)")
    logger.info(f"Feature selection: Top {diff_percentile*100:.1f}% difference")
    logger.info(f"TICA Lag time: {lag_time} frames")
    logger.info(f"Target TICA dimensions: {n_dim}")
    logger.info(f"Plot type: {plot_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output log file: {output_log_file}")

    # Update output filenames
    plot_filename = f"{output_basename}_backbone_torsions{percentile_tag}_tica_lag{lag_time}_dim{n_dim}_plot.png"
    tica_model_filename = f"{output_basename}_backbone_torsions{percentile_tag}_tica_lag{lag_time}_dim{n_dim}_model.joblib"
    selected_indices_filename = f"{output_basename}_backbone_torsions{percentile_tag}_selected_feature_indices.npy"
    selected_features_filename = f"{output_basename}_backbone_torsions{percentile_tag}_selected_features.npy"
    tica_output_filename = f"{output_basename}_backbone_torsions{percentile_tag}_tica_output_lag{lag_time}_dim{n_dim}.npy"

    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_model_file = os.path.join(output_dir, tica_model_filename)
    output_selected_indices_file = os.path.join(output_dir, selected_indices_filename)
    output_selected_features_file = os.path.join(output_dir, selected_features_filename)
    output_tica_output_file = os.path.join(output_dir, tica_output_filename)

    logger.info(f"Output Selected Indices file: {output_selected_indices_file}")
    logger.info(f"Output Selected Features file: {output_selected_features_file}")
    logger.info(f"Output TICA Model file: {output_tica_model_file}")
    logger.info(f"Output TICA Projection file: {output_tica_output_file}")
    logger.info(f"Output Plot file: {output_plot_file}")

    tica_model = None

    try:
        # 1. Load Structures
        logger.info("Loading structures...")
        try:
            traj_full = md.load(traj_pdb_file)
            traj_start = md.load(start_pdb_file)
            traj_end = md.load(end_pdb_file)
            if traj_full.n_frames == 0: raise ValueError(f"Trajectory PDB {traj_pdb_file} has 0 frames.")
            if traj_start.n_frames == 0: raise ValueError(f"Start PDB {start_pdb_file} has 0 frames.")
            if traj_end.n_frames == 0: raise ValueError(f"End PDB {end_pdb_file} has 0 frames.")
            # Use only the first frame for start/end references
            traj_start = traj_start[0]
            traj_end = traj_end[0]
            logger.info(f"Loaded trajectory ({traj_full.n_frames} frames), start ({traj_start.n_frames} frame), end ({traj_end.n_frames} frame).")
        except Exception as e:
             logger.error(f"Error loading PDB files: {e}")
             sys.exit(1)

        n_frames = traj_full.n_frames
        if n_frames <= lag_time:
            logger.warning(f"Warning: Trajectory length ({n_frames}) <= lag time ({lag_time}). TICA requires n_frames > lag_time.")
            if n_frames <= 1: raise ValueError("Cannot compute TICA with <= 1 frame.")


        # 2. Calculate ALL Features for all structures
        logger.info("Calculating all backbone torsion features...")
        features_full, _ = calculate_features(traj_full, logger)
        features_start, _ = calculate_features(traj_start, logger)
        features_end, _ = calculate_features(traj_end, logger)
        n_total_features = features_full.shape[1]
        logger.info(f"Full feature space dimension: {n_total_features}")


        # 3. Feature Selection based on Difference
        logger.info("Selecting features based on difference between start and end states...")
        if features_start.shape[1] != n_total_features or features_end.shape[1] != n_total_features:
            raise ValueError("Feature dimension mismatch between full trajectory and start/end states.")

        n_angle_pairs = n_total_features // 2 # Assuming paired cos/sin for each angle
        if n_total_features % 2 != 0:
            logger.warning("Warning: Total number of features is odd. Assuming last feature is unpaired?")
            # Adjust logic if unpaired features are possible/expected

        feature_diff_robust = np.zeros(n_angle_pairs)
        for i in range(n_angle_pairs):
            idx_cos = 2 * i      # Index for cos(angle_i)
            idx_sin = 2 * i + 1  # Index for sin(angle_i)
            cos1, sin1 = features_start[0, idx_cos], features_start[0, idx_sin]
            cos2, sin2 = features_end[0, idx_cos], features_end[0, idx_sin]
            feature_diff_robust[i] = (sin1 - sin2)**2 + (cos1 - cos2)**2 # Squared Euclidean dist in (cos, sin)

        # Determine number of features to keep
        num_pairs_to_keep_requested = int(np.ceil(n_angle_pairs * diff_percentile))
        num_sincos_to_keep_requested = 2 * num_pairs_to_keep_requested

        # === Dimensionality Constraint ===
        # Max features = T-1, ensure we don't exceed this
        max_features_tica = max(1, n_frames - 1) # Must be at least 1
        if num_sincos_to_keep_requested > max_features_tica:
             logger.warning(f"Requested {num_sincos_to_keep_requested} features ({diff_percentile*100:.1f}% of pairs) but trajectory length ({n_frames}) limits TICA input dim to {max_features_tica}. Capping selected features.")
             num_sincos_to_keep = max_features_tica
             # Recalculate pairs to keep based on capped sin/cos count
             num_pairs_to_keep = num_sincos_to_keep // 2 # Integer division, might lose one if max_features_tica is odd
             # Ensure an even number for pairs, if possible and necessary
             if num_sincos_to_keep % 2 != 0 and num_pairs_to_keep > 0: # If odd limit and we can reduce
                 num_pairs_to_keep -= 1
             num_sincos_to_keep = 2 * num_pairs_to_keep # Final even number <= max_features_tica
        else:
            num_pairs_to_keep = num_pairs_to_keep_requested
            num_sincos_to_keep = num_sincos_to_keep_requested

        logger.info(f"Targeting {num_pairs_to_keep} angle pairs ({num_sincos_to_keep} sin/cos features) based on difference and frame limit.")

        if num_sincos_to_keep < 1:
            msg = f"Calculated number of features to keep is less than 1 ({num_sincos_to_keep}). Cannot proceed."
            logger.error(msg)
            raise ValueError(msg)

        # Select top K pairs and their corresponding sin/cos indices
        indices_sorted_by_diff = np.argsort(feature_diff_robust)[::-1] # Descending order
        indices_top_k_pairs = indices_sorted_by_diff[:num_pairs_to_keep]

        selected_indices = [] # Indices of columns to keep from the original N_total_features
        for pair_idx in indices_top_k_pairs:
            idx_cos = 2 * pair_idx
            idx_sin = 2 * pair_idx + 1
            selected_indices.extend([idx_cos, idx_sin])
        selected_indices = sorted(selected_indices) # Keep original relative order

        # Ensure the final selected indices count matches num_sincos_to_keep
        if len(selected_indices) != num_sincos_to_keep:
             logger.warning(f"Mismatch after selecting indices: Expected {num_sincos_to_keep}, got {len(selected_indices)}. This might happen if pair calculation logic needs adjustement for odd limits.")
             # Use the actual length for selection
             num_sincos_to_keep = len(selected_indices)

        if not selected_indices:
            msg = "No features selected after applying difference criteria and limits. Cannot proceed."
            logger.error(msg)
            raise ValueError(msg)

        # Save the indices of the selected features
        try:
            logger.info(f"Saving indices of the {len(selected_indices)} selected features to '{output_selected_indices_file}'...")
            np.save(output_selected_indices_file, np.array(selected_indices))
            logger.info("Selected feature indices saved.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save selected feature indices. Details: {e}")

        # Filter the full trajectory features
        features_selected = features_full[:, selected_indices]
        logger.info(f"Feature selection complete. Selected feature shape: {features_selected.shape}")

        # Save the selected features data
        try:
            logger.info(f"Saving selected features data to '{output_selected_features_file}'...")
            np.save(output_selected_features_file, features_selected)
            logger.info("Selected features data saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save selected features data. Details: {e}")


        # 4. Perform TICA on the Selected Features
        logger.info(f"Performing TICA calculation on {features_selected.shape[1]} selected features (lag={lag_time})...")
        tica_input_dim = features_selected.shape[1]

        # Final check on target TICA dimension vs selected feature dimension
        if n_dim > tica_input_dim:
            logger.warning(f"Warning: Requested TICA dim ({n_dim}) > selected features dim ({tica_input_dim}). Reducing TICA dim to {tica_input_dim}.")
            n_dim = tica_input_dim
        if n_dim < 1: # Safety check
             n_dim = 1

        tica_estimator = TICA(lagtime=lag_time, dim=n_dim, scaling='kinetic_map')
        try:
            logger.info(f"Fitting TICA model on data shape {features_selected.shape}...")
            tica_estimator.fit(features_selected)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"Error during TICA fitting on selected features (lag={lag_time}, dim={n_dim}): {e}")
            sys.exit(1)

        tica_model = tica_estimator.fetch_model()
        if tica_model is None:
            logger.error("Failed to fetch TICA model after fitting.")
            sys.exit(1)

        logger.info("Transforming selected features using TICA model...")
        tica_output = tica_model.transform(features_selected)

        actual_dims_computed = tica_output.shape[1]
        logger.info(f"TICA calculation complete. Output projection shape: {tica_output.shape}")

        if actual_dims_computed == 0: sys.exit("Error: TICA resulted in 0 dimensions.")
        if actual_dims_computed < n_dim:
             logger.warning(f"TICA computed {actual_dims_computed} dimensions, less than final target {n_dim}.")
             n_dim_plot = actual_dims_computed
        else:
            n_dim_plot = n_dim

        # Log Timescales
        try:
            computed_timescales = tica_model.timescales(lagtime=lag_time)
            logger.info(f"TICA Timescales (lag={lag_time} K): {computed_timescales[:n_dim_plot]}")
        except Exception as e:
            logger.warning(f"Could not compute timescales: {e}")

        # Save TICA Output and Model
        try:
            logger.info(f"Saving TICA projection data to '{output_tica_output_file}'...")
            np.save(output_tica_output_file, tica_output)
        except Exception as e: logger.warning(f"Failed to save TICA projection: {e}")

        try:
            logger.info(f"Saving TICA model object to '{output_tica_model_file}'...")
            joblib.dump(tica_model, output_tica_model_file)
            logger.info("TICA model saved.")
        except Exception as e: logger.error(f"Failed to save TICA model: {e}"); sys.exit(1)


        # 5. Generate Plot
        logger.info(f"Generating TICA plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dim = min(2, n_dim_plot)
        title = f'TICA (Top {diff_percentile*100:.0f}% Diff feats -> {tica_input_dim}D, Lag={lag_time})'

        if plot_dim == 1:
            # ... (Plot 1D) ...
            x_plot = np.arange(len(tica_output))
            y_plot = tica_output[:, 0]
            ax.plot(x_plot, y_plot, marker='.', linestyle='-', markersize=2, alpha=0.7)
            ax.set_xlabel('Frame Index'); ax.set_ylabel('TICA Component 1')
        elif plot_dim >= 2:
             # ... (Plot 2D) ...
            x_plot = tica_output[:, 0]; y_plot = tica_output[:, 1]
            ax.set_xlabel('TICA Component 1 (IC1)'); ax.set_ylabel('TICA Component 2 (IC2)')
            if plot_type == 'scatter':
                scatter = ax.scatter(x_plot, y_plot, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Frame Index')
            elif plot_type == 'hist2d':
                try:
                    from matplotlib.colors import LogNorm
                    if np.isclose(np.var(x_plot), 0) or np.isclose(np.var(y_plot), 0):
                         scatter = ax.scatter(x_plot, y_plot, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                         cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Frame Index'); title += " (hist2d->scatter)"
                         logger.warning("Hist2d fallback to scatter due to low variance.")
                    else:
                         counts, xedges, yedges, image = ax.hist2d(x_plot, y_plot, bins=100, cmap='viridis', cmin=1, norm=LogNorm())
                         cbar = fig.colorbar(image, ax=ax); cbar.set_label('Counts (log scale)')
                except ValueError as e:
                     scatter = ax.scatter(x_plot, y_plot, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                     cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Frame Index'); title += " (hist2d->scatter)"
                     logger.warning(f"Hist2d fallback to scatter: {e}.")

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(output_plot_file, dpi=300)
        logger.info(f"TICA plot saved to '{output_plot_file}'")
        plt.close(fig)

    # --- Error Handling ---
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except MemoryError:
        logger.error(f"Error: Insufficient memory.")
        sys.exit(1)
    except (ValueError, TypeError, np.linalg.LinAlgError, RuntimeError) as e:
         logger.error(f"Error during analysis: {e}")
         import traceback
         logger.debug(traceback.format_exc())
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logging.shutdown()

    print("--- Difference-Selected TICA Analysis Finished ---")
    return tica_model

def main():
    parser = argparse.ArgumentParser(
        description='Perform TICA using backbone torsion features selected by difference between start/end states.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Inputs
    parser.add_argument('traj_pdb_file', help='Path to the input multi-frame PDB trajectory.')
    parser.add_argument('start_pdb_file', help='Path to the single-frame start reference PDB.')
    parser.add_argument('end_pdb_file', help='Path to the single-frame end reference PDB.')
    # Feature Selection Parameter
    parser.add_argument(
        '--diff-percentile', type=float, default=0.2,
        help='Keep the top X percent of backbone angle pairs showing the largest difference between start/end states (e.g., 0.2 for 20%%).'
    )
    # TICA Parameters
    parser.add_argument('-l', '--lag', type=int, default=10, help='TICA lag time (frames).')
    parser.add_argument('-d', '--dim', type=int, default=2, help='Target TICA dimensions.')
    # Outputs
    parser.add_argument('-o', '--output-dir', default="diff_tica_output", help='Directory for outputs.')
    parser.add_argument('--plot-type', choices=['scatter', 'hist2d'], default='scatter', help='Type of plot.')
    parser.add_argument('--basename', default=None, help='Basename for outputs (default: derived from traj PDB).')
    parser.add_argument('--v', '--verbose', action='store_true', help='Enable verbose logging.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.isfile(args.traj_pdb_file): sys.exit(f"Error: Trajectory PDB not found: {args.traj_pdb_file}")
    if not os.path.isfile(args.start_pdb_file): sys.exit(f"Error: Start PDB not found: {args.start_pdb_file}")
    if not os.path.isfile(args.end_pdb_file): sys.exit(f"Error: End PDB not found: {args.end_pdb_file}")
    if not (0 < args.diff_percentile <= 1.0): sys.exit("Error: --diff-percentile must be between 0 (exclusive) and 1 (inclusive).")
    if args.lag <= 0: sys.exit("Error: --lag must be positive.")
    if args.dim < 1: sys.exit("Error: --dim must be >= 1.")

    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(args.traj_pdb_file))[0]

    try: os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e: sys.exit(f"Error creating output dir '{args.output_dir}': {e}")

    # Setup logger based on verbosity AFTER potential errors above
    logger = setup_logging(os.path.join(args.output_dir, f"{args.basename}_diffsel_tica.log"), args.v)

    run_diff_tica_analysis(
        traj_pdb_file=args.traj_pdb_file,
        start_pdb_file=args.start_pdb_file,
        end_pdb_file=args.end_pdb_file,
        diff_percentile=args.diff_percentile,
        lag_time=args.lag,
        n_dim=args.dim,
        output_dir=args.output_dir,
        output_basename=args.basename,
        plot_type=args.plot_type
    )

if __name__ == "__main__":
    main()
