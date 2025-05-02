# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import sys
import os
import warnings
import logging
from itertools import combinations

warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
# Keep the RuntimeWarning filter for numpy/matplotlib potentially triggered during TICA/plotting
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import mdtraj as md
    import deeptime
    from deeptime.decomposition import TICA
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib # Using joblib for saving the deeptime model object
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"Error: Missing required library. Please install deeptime, MDTraj, Matplotlib, NumPy, joblib, argparse.")
    logging.error(f"Run: pip install deeptime mdtraj matplotlib numpy joblib argparse")
    logging.error(f"Import error details: {e}")
    sys.exit(1)

def setup_logging(log_file):
    """Configure logger."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    # Remove existing handlers if any
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    return root_logger

def calculate_features(traj, feature_type, logger):
    """
    Calculates features directly using MDTraj.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.
    feature_type : str
        Type of features ('ca_dist', 'heavy_atom', 'backbone_torsions').
    logger : logging.Logger
        The logger instance.

    Returns
    -------
    features_data : np.ndarray or None
        The calculated features, or None if an error occurred.
    needs_alignment : bool
        Whether the 'heavy_atom' feature type requires alignment.
    """
    topology = traj.topology
    features_data = None
    needs_alignment = False
    feature_dim = 0

    if feature_type == 'ca_dist':
        ca_indices = topology.select('name CA and protein')
        if len(ca_indices) < 2:
            msg = f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found) for distance calculation."
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Selected {len(ca_indices)} C-alpha atoms for distance calculation.")
        # Create pairs of indices for distance calculation
        ca_pairs = list(combinations(ca_indices, 2))
        if not ca_pairs:
             msg = f"Could not generate any C-alpha pairs for distance calculation."
             logger.error(msg)
             raise ValueError(msg)
        logger.info(f"Calculating {len(ca_pairs)} pairwise C-alpha distances...")
        features_data = md.compute_distances(traj, ca_pairs, periodic=False) # Assuming non-periodic common for TICA
        feature_dim = features_data.shape[1]
        logger.info(f"Using C-alpha distances. Feature dimension: {feature_dim}")

    elif feature_type == 'heavy_atom':
        heavy_atom_indices = topology.select('not element H and protein')
        if len(heavy_atom_indices) == 0:
             msg = "No heavy atoms found in protein residues."
             logger.error(msg)
             raise ValueError(msg)
        logger.info(f"Selected {len(heavy_atom_indices)} heavy atoms.")
        # Alignment will be done *before* calling this function if needed.
        # Here we just extract the coordinates.
        logger.info("Extracting heavy atom coordinates...")
        # Reshape coordinates from (n_frames, n_atoms, 3) to (n_frames, n_atoms * 3)
        features_data = traj.xyz[:, heavy_atom_indices, :].reshape(traj.n_frames, -1)
        feature_dim = features_data.shape[1]
        needs_alignment = True # Signal that alignment is expected for this feature type
        logger.info(f"Using heavy atom coordinates. Feature dimension: {feature_dim}")

    elif feature_type == 'backbone_torsions':
        logger.info("Calculating backbone torsion angles (phi, psi)...")
        phi_indices, phi_angles = md.compute_phi(traj, periodic=False)
        psi_indices, psi_angles = md.compute_psi(traj, periodic=False)

        # Handle cases where phi/psi might not be computable for all residues (e.g., terminals)
        # MDTraj returns NaN for these. We convert sin(NaN)/cos(NaN) to 0.0.
        phi_cos = np.cos(phi_angles)
        phi_sin = np.sin(phi_angles)
        psi_cos = np.cos(psi_angles)
        psi_sin = np.sin(psi_angles)

        # Replace NaNs resulting from angle calculation or cos/sin with 0.0
        phi_cos = np.nan_to_num(phi_cos, nan=0.0)
        phi_sin = np.nan_to_num(phi_sin, nan=0.0)
        psi_cos = np.nan_to_num(psi_cos, nan=0.0)
        psi_sin = np.nan_to_num(psi_sin, nan=0.0)

        features_data = np.hstack([phi_cos, phi_sin, psi_cos, psi_sin])
        feature_dim = features_data.shape[1]

        logger.info(f"Using backbone torsions (cos/sin of phi/psi). Feature dimension: {feature_dim}")
        if feature_dim == 0:
            msg = "Could not calculate any backbone torsions (dimension is 0). Is it a valid protein structure with multiple residues?"
            logger.error(msg)
            raise ValueError(msg)

    else:
         msg = f"Invalid feature type '{feature_type}' specified."
         logger.error(msg)
         raise ValueError(msg)

    if features_data is None or features_data.shape[0] == 0 or features_data.shape[1] == 0:
         msg = f"Feature calculation for '{feature_type}' resulted in empty or zero-dimension data."
         logger.error(msg)
         raise ValueError(msg)

    return features_data, needs_alignment


def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type, plot_type):
    """
    Performs TICA analysis on a multi-frame PDB file using deeptime,
    saves the TICA model, projection plot, intermediate feature files, and logs
    to the specified directory.

    Parameters are the same as the original script, substituting deeptime for PyEMMA.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}.log"
    output_log_file = os.path.join(output_dir, log_filename)
    logger = setup_logging(output_log_file)

    logger.info("--- Starting TICA Analysis (using deeptime) ---")
    logger.info(f"Deeptime version: {deeptime.__version__}")
    logger.info(f"MDTraj version: {md.__version__}")
    logger.info(f"Input PDB: {pdb_file}")
    logger.info(f"Lag time: {lag_time} frames")
    logger.info(f"Target TICA dimensions: {n_dim}")
    logger.info(f"Feature type: {feature_type}")
    logger.info(f"Plot type: {plot_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output log file: {output_log_file}")

    # Update output filenames, especially for the model
    plot_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}_plot.png"
    tica_model_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}_model.joblib" # Using joblib extension
    features_data_filename = f"{output_basename}_features_{feature_type}.npy"
    tica_output_filename = f"{output_basename}_tica_output_lag{lag_time}_dim{n_dim}_{feature_type}.npy"

    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_model_file = os.path.join(output_dir, tica_model_filename)
    output_features_data_file = os.path.join(output_dir, features_data_filename)
    output_tica_output_file = os.path.join(output_dir, tica_output_filename)

    logger.info(f"Output plot file: {output_plot_file}")
    logger.info(f"Output TICA model file: {output_tica_model_file}")
    logger.info(f"Output Features data file: {output_features_data_file}")
    logger.info(f"Output TICA projection file: {output_tica_output_file}")

    tica_model = None # Changed from tica_obj

    if not os.path.exists(pdb_file):
        logger.error(f"Error: Input PDB file not found at '{pdb_file}'")
        sys.exit(1)

    try:
        logger.info("Loading trajectory...")
        # Load entire trajectory at once. Consider md.iterload for very large files if memory becomes an issue.
        try:
            # Load topology from first frame for consistency, then load full traj
            # topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology # Not robust for all PDB variants
            traj = md.load(pdb_file) # MDTraj usually infers topology well from multi-frame PDBs
            if traj is None or traj.n_frames == 0 or traj.n_atoms == 0:
                msg = f"Failed to load a valid trajectory from {pdb_file}. Check PDB format and content."
                logger.error(msg)
                raise ValueError(msg)
            topology_mdtraj = traj.topology # Get topology from loaded traj
            logger.info(f"Loaded trajectory: {traj.n_frames} frames, {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")

        except Exception as e:
             logger.error(f"Error: Failed to load trajectory from {pdb_file}. Check format/content. Details: {e}")
             sys.exit(1)


        if traj.n_frames <= lag_time:
            logger.warning(f"Warning: Trajectory length ({traj.n_frames}) is not greater than lag time ({lag_time}). TICA requires n_frames > lag_time.")
            if traj.n_frames <= 1:
                 msg = "Cannot compute TICA with <= 1 frame."
                 logger.error(msg)
                 raise ValueError(msg)
            # Let deeptime handle the error if n_frames == lag_time, but log a strong warning.
            if traj.n_frames == lag_time:
                logger.warning(f"TICA calculation might fail or be unstable as n_frames == lag_time.")


        logger.info(f"Calculating features ({feature_type})...")
        # Calculate features first, then align if needed
        features_data, needs_alignment = calculate_features(traj, feature_type, logger)
        logger.info(f"Feature calculation complete. Feature shape: {features_data.shape}")


        if needs_alignment:
            if traj.n_frames < 2:
                 logger.warning("Warning: Only 1 frame in trajectory, cannot perform alignment (needed for heavy_atom features).")
            else:
                 logger.info("Aligning trajectory to the first frame (using heavy atoms)...")
                 heavy_atom_indices_align = topology_mdtraj.select('not element H and protein')
                 if len(heavy_atom_indices_align) == 0:
                      logger.warning("Warning: No heavy atoms found for alignment, proceeding without alignment.")
                 else:
                    try:
                        traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices_align, ref_atom_indices=heavy_atom_indices_align)
                        logger.info("Alignment complete.")
                        # Recalculate features *after* alignment for heavy atoms
                        if feature_type == 'heavy_atom':
                            logger.info("Re-calculating heavy atom coordinates after alignment...")
                            features_data, _ = calculate_features(traj, feature_type, logger) # Needs_alignment flag not needed here
                            logger.info(f"Recalculated feature shape: {features_data.shape}")
                    except Exception as e:
                        logger.error(f"Error during trajectory alignment: {e}. Proceeding with unaligned coordinates for heavy atoms, results may be affected.")


        # --- Log summary statistics of the features ---
        if features_data.size > 0:
            logger.info(f"Feature data summary (first 5 features if >5):")
            num_features_to_log = min(5, features_data.shape[1])
            logger.info(f"  Shape: {features_data.shape}")
            try:
                # Ensure data is finite for stats
                finite_features = features_data[np.isfinite(features_data).all(axis=1)]
                if finite_features.shape[0] > 0:
                    mean_vals = np.mean(finite_features[:, :num_features_to_log], axis=0)
                    std_vals = np.std(finite_features[:, :num_features_to_log], axis=0)
                    min_vals = np.min(finite_features[:, :num_features_to_log], axis=0)
                    max_vals = np.max(finite_features[:, :num_features_to_log], axis=0)
                    logger.info(f"  Mean (first {num_features_to_log}, finite data only): {mean_vals}")
                    logger.info(f"  Std Dev (first {num_features_to_log}, finite data only): {std_vals}")
                    logger.info(f"  Min (first {num_features_to_log}, finite data only): {min_vals}")
                    logger.info(f"  Max (first {num_features_to_log}, finite data only): {max_vals}")
                else:
                     logger.warning("Feature data contains NaNs or Infs, cannot compute summary statistics.")
            except Exception as stat_err:
                logger.warning(f"Could not compute summary statistics for features: {stat_err}")
        else:
            logger.warning("Feature data is empty, cannot compute summary statistics.")
        # --- End logging summary statistics ---

        try:
            logger.info(f"Saving features data to '{output_features_data_file}'...")
            np.save(output_features_data_file, features_data)
            logger.info("Features data saved successfully.")
            logger.info(f"To inspect the full features data, load with: `import numpy as np; features = np.load('{output_features_data_file}')`")
        except Exception as e:
            logger.warning(f"Warning: Failed to save features data to '{output_features_data_file}'. Continuing analysis. Details: {e}")


        logger.info(f"Performing TICA calculation with lag={lag_time} using deeptime...")
        actual_n_frames = features_data.shape[0]
        feature_dim = features_data.shape[1]

        # Check dimension feasibility
        if actual_n_frames <= lag_time:
             # This case should have been caught earlier, but double-check
             msg = f"Cannot compute TICA: Trajectory length ({actual_n_frames}) <= lag time ({lag_time})."
             logger.error(msg)
             raise ValueError(msg)

        if n_dim > feature_dim:
            logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) exceeds feature dimension ({feature_dim}). Reducing to {feature_dim}.")
            n_dim = feature_dim
        elif n_dim <= 0:
             logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) must be >= 1. Setting to 1.")
             n_dim = 1

        # Check for NaNs/Infs in features which will break TICA
        if not np.all(np.isfinite(features_data)):
             num_nonfinite = np.sum(~np.isfinite(features_data))
             msg = f"Feature data contains {num_nonfinite} non-finite values (NaN or Inf). TICA cannot proceed. Check feature calculation method ('{feature_type}')."
             logger.error(msg)
             raise ValueError(msg)


        # Instantiate and fit deeptime TICA estimator
        # scaling='kinetic_map' is default, but being explicit is fine.
        # epsilon handles rank deficiency internally.
        tica_estimator = TICA(lagtime=lag_time, dim=n_dim, scaling='kinetic_map')

        # Fit the model
        try:
            tica_estimator.fit(features_data)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"Error during TICA fitting (lag={lag_time}, dim={n_dim}): {e}")
            logger.error("This might be due to insufficient data variability for the given lag time, rank deficiency not handled by epsilon, or issues with feature data.")
            sys.exit(1)

        # Fetch the trained model
        tica_model = tica_estimator.fetch_model()
        if tica_model is None:
            logger.error("Failed to fetch TICA model after fitting.")
            sys.exit(1)

        # Transform the data (get the projection)
        tica_output = tica_model.transform(features_data)

        actual_dims_computed = tica_output.shape[1]
        if actual_dims_computed == 0:
             # This shouldn't happen if fit succeeded and dim>=1, but check anyway
             logger.error(f"TICA computation resulted in 0 dimensions despite requesting {n_dim}. Aborting.")
             sys.exit(1)
        if actual_dims_computed < n_dim:
             logger.warning(f"TICA computation resulted in {actual_dims_computed} dimensions, which is less than the requested {n_dim}. This might indicate rank deficiency.")
             n_dim = actual_dims_computed # Update n_dim to actual for plotting


        logger.info(f"TICA calculation complete. Output projection shape: {tica_output.shape}")
        # Access eigenvalues and timescales (note: timescales is a method)
        try:
            # Pass the *same* lag time used for estimation to timescales()
            computed_timescales = tica_model.timescales(lagtime=lag_time)
            logger.info(f"TICA Timescales (lag={lag_time}): {computed_timescales}")
        except Exception as e:
            logger.warning(f"Could not compute timescales: {e}") # Might happen if eigenvalues are complex or <= 0

        try:
            logger.info(f"Saving TICA projection data to '{output_tica_output_file}'...")
            np.save(output_tica_output_file, tica_output)
            logger.info("TICA projection data saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save TICA projection data to '{output_tica_output_file}'. Continuing analysis. Details: {e}")

        try:
            logger.info(f"Saving TICA model object to '{output_tica_model_file}'...")
            joblib.dump(tica_model, output_tica_model_file)
            logger.info("TICA model object saved successfully.")
            logger.info(f"To load the model later: `import joblib; model = joblib.load('{output_tica_model_file}')`")
        except Exception as e:
            # Saving the model is crucial for reuse, treat failure as error.
            logger.error(f"Error: Failed to save TICA model to '{output_tica_model_file}'. Cannot continue. Details: {e}")
            sys.exit(1)

        logger.info(f"Generating TICA plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use the potentially updated n_dim (actual_dims_computed)
        plot_dim = min(2, n_dim)
        plot_title_base = f'TICA Projection (Lag: {lag_time} frames, Features: {feature_type})'


        if plot_dim == 0:
            logger.warning("Warning: TICA resulted in 0 dimensions. Cannot generate plot.") # Should be caught earlier
        elif plot_dim == 1:
            logger.info("Plotting TICA Component 1 vs Frame Index.")
            x = np.arange(len(tica_output))
            y = tica_output[:, 0]
            xlabel = 'Frame Index'
            ylabel = 'TICA Component 1'
            plot_title = plot_title_base
            ax.plot(x, y, marker='.', linestyle='-', markersize=2, alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(output_plot_file, dpi=300)
            logger.info(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig)

        elif plot_dim >= 2: # Plot first two components if available
            x = tica_output[:, 0]
            y = tica_output[:, 1]
            xlabel = 'TICA Component 1 (IC1)'
            ylabel = 'TICA Component 2 (IC2)'
            plot_title = plot_title_base

            if plot_type == 'scatter':
                scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Frame Index')
            elif plot_type == 'hist2d':
                try:
                    from matplotlib.colors import LogNorm
                    # Check for variance before attempting hist2d
                    if np.isclose(np.var(x), 0) or np.isclose(np.var(y), 0):
                        logger.warning("Warning: Very low variance in TICA components. Hist2d may fail or be uninformative. Falling back to scatter plot.")
                        scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label('Frame Index')
                        plot_title += " (hist2d failed due to low variance, showing scatter)"
                    else:
                        counts, xedges, yedges, image = ax.hist2d(
                            x, y, bins=100, cmap='viridis', cmin=1 # cmin=1 avoids issues with empty bins
                        )
                        cbar = fig.colorbar(image, ax=ax)
                        cbar.set_label('Counts')
                except ValueError as e:
                    logger.warning(f"Warning: hist2d plot failed: {e}. Check data range/variance. Falling back to scatter plot.")
                    scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Frame Index')
                    plot_title += " (hist2d failed, showing scatter)"
            else:
                # This case should be caught by argparse choices, but check defensively
                msg = f"Invalid plot_type: {plot_type}"
                logger.error(msg)
                raise ValueError(msg)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(output_plot_file, dpi=300)
            logger.info(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig)


    except FileNotFoundError:
        # Error already logged if pdb_file not found
        pass
    except MemoryError:
        logger.error(f"Error: Insufficient memory during TICA analysis. Consider using fewer features, downsampling data, or investigating memory usage.")
        sys.exit(1)
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
         # Catch common calculation/setup errors
         logger.error(f"Error during TICA analysis setup or calculation: {e}")
         import traceback
         logger.debug(traceback.format_exc()) # Log traceback for debugging if needed
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during TICA analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Ensure logging handlers are closed properly
        logging.shutdown()

    print("--- TICA Analysis Finished ---") # Keep print statement for basic console feedback
    return tica_model # Return the deeptime model object

def main():
    parser = argparse.ArgumentParser(
        description='Perform TICA on a PDB trajectory using deeptime, save the TICA model, projection plot, intermediate files, and logs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pdb_file',
        help='Path to the input multi-frame PDB file (trajectory).'
    )
    parser.add_argument(
        '-l', '--lag', type=int, default=10,
        help='TICA lag time (in frames). Must be > 0.'
    )
    parser.add_argument(
        '-d', '--dim', type=int, default=2,
        help='Number of TICA dimensions to calculate (>= 1).'
    )
    parser.add_argument(
        '-o', '--output-dir', default="tica_analysis_output_deeptime", # Changed default dir name slightly
        help='Directory to save the TICA projection plot, TICA model file, intermediate files, and log.'
    )
    parser.add_argument(
        '--features', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'],
        default='backbone_torsions',
        help='Type of features to use for TICA. "backbone_torsions" is often recommended.'
    )
    parser.add_argument(
        '--plot-type', choices=['scatter', 'hist2d'], default='scatter',
        help='Type of plot for TICA projection (first 2 components). "scatter" is better for paths/sparse data.'
    )
    parser.add_argument(
        '--basename', default=None,
        help='Basename for output files. Defaults to input PDB filename without extension.'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.isfile(args.pdb_file):
         print(f"Error: Input PDB file not found: {args.pdb_file}", file=sys.stderr)
         sys.exit(1)
    if args.lag <= 0:
        print("Error: Lag time (--lag) must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.dim < 1:
        print("Error: TICA dimensions (--dim) must be >= 1.", file=sys.stderr)
        sys.exit(1)
    # --- End Input Validation ---


    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(args.pdb_file))[0]

    # Create the full path for the output directory early
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)


    run_tica_analysis(
        pdb_file=args.pdb_file,
        lag_time=args.lag,
        n_dim=args.dim,
        output_dir=args.output_dir,
        output_basename=args.basename,
        feature_type=args.features,
        plot_type=args.plot_type
    )

if __name__ == "__main__":
    main()
