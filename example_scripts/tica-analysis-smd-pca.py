#!/usr/bin/env python3

import argparse
import sys
import os
import warnings
import logging
from itertools import combinations

warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning) # Keep for numpy/matplotlib
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore potential sklearn/numpy warnings

try:
    import mdtraj as md
    import deeptime
    from deeptime.decomposition import TICA
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib # Using joblib for saving model objects
    from sklearn.decomposition import PCA # Import PCA
    import sklearn
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"Error: Missing required library. Please install deeptime, MDTraj, Matplotlib, NumPy, joblib, scikit-learn, argparse.")
    logging.error(f"Run: pip install deeptime mdtraj matplotlib numpy joblib scikit-learn argparse")
    logging.error(f"Import error details: {e}")
    sys.exit(1)

# --- setup_logging and calculate_features remain the same as previous deeptime version ---
def setup_logging(log_file):
    """Configure logger."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
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
    """Calculates features directly using MDTraj."""
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
        ca_pairs = list(combinations(ca_indices, 2))
        if not ca_pairs:
             msg = f"Could not generate any C-alpha pairs for distance calculation."
             logger.error(msg)
             raise ValueError(msg)
        logger.info(f"Calculating {len(ca_pairs)} pairwise C-alpha distances...")
        features_data = md.compute_distances(traj, ca_pairs, periodic=False)
        feature_dim = features_data.shape[1]
        logger.info(f"Using C-alpha distances. Feature dimension: {feature_dim}")

    elif feature_type == 'heavy_atom':
        heavy_atom_indices = topology.select('not element H and protein')
        if len(heavy_atom_indices) == 0:
             msg = "No heavy atoms found in protein residues."
             logger.error(msg)
             raise ValueError(msg)
        logger.info(f"Selected {len(heavy_atom_indices)} heavy atoms.")
        logger.info("Extracting heavy atom coordinates...")
        features_data = traj.xyz[:, heavy_atom_indices, :].reshape(traj.n_frames, -1)
        feature_dim = features_data.shape[1]
        needs_alignment = True
        logger.info(f"Using heavy atom coordinates. Feature dimension: {feature_dim}")

    elif feature_type == 'backbone_torsions':
        logger.info("Calculating backbone torsion angles (phi, psi)...")
        phi_indices, phi_angles = md.compute_phi(traj, periodic=False)
        psi_indices, psi_angles = md.compute_psi(traj, periodic=False)
        phi_cos = np.cos(phi_angles)
        phi_sin = np.sin(phi_angles)
        psi_cos = np.cos(psi_angles)
        psi_sin = np.sin(psi_angles)
        phi_cos = np.nan_to_num(phi_cos, nan=0.0)
        phi_sin = np.nan_to_num(phi_sin, nan=0.0)
        psi_cos = np.nan_to_num(psi_cos, nan=0.0)
        psi_sin = np.nan_to_num(psi_sin, nan=0.0)
        features_data = np.hstack([phi_cos, phi_sin, psi_cos, psi_sin])
        if features_data.size == 0: # Check if hstack resulted in empty array
            msg = "Backbone torsion calculation resulted in empty feature array. Is it a valid protein?"
            logger.error(msg)
            raise ValueError(msg)
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
# --- End of unchanged functions ---


def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type, plot_type, pca_variance_cutoff):
    """
    Performs optional PCA and then TICA analysis on a multi-frame PDB file using deeptime,
    saves models, projections, intermediate files, and logs.

    Parameters:
        pdb_file (str): Path to input PDB.
        lag_time (int): TICA lag time.
        n_dim (int): Target TICA dimensions.
        output_dir (str): Directory for output.
        output_basename (str): Basename for output files.
        feature_type (str): Feature type ('ca_dist', 'heavy_atom', 'backbone_torsions').
        plot_type (str): Plot type ('scatter', 'hist2d').
        pca_variance_cutoff (float): Variance cutoff for PCA (e.g., 0.95). <= 0 disables PCA.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add PCA info to log/output filenames if used
    pca_tag = f"_pca{int(pca_variance_cutoff*100)}" if pca_variance_cutoff > 0 else ""
    log_filename = f"{output_basename}_{feature_type}{pca_tag}_tica_lag{lag_time}_dim{n_dim}.log"
    output_log_file = os.path.join(output_dir, log_filename)
    logger = setup_logging(output_log_file)

    logger.info("--- Starting PCA-TICA Analysis (using scikit-learn & deeptime) ---")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"Deeptime version: {deeptime.__version__}")
    logger.info(f"MDTraj version: {md.__version__}")
    logger.info(f"Input PDB: {pdb_file}")
    logger.info(f"Feature type: {feature_type}")
    if pca_variance_cutoff > 0:
        logger.info(f"PCA pre-processing: Enabled (Variance Cutoff: {pca_variance_cutoff:.2f})")
    else:
        logger.info(f"PCA pre-processing: Disabled")
    logger.info(f"TICA Lag time: {lag_time} frames")
    logger.info(f"Target TICA dimensions: {n_dim}")
    logger.info(f"Plot type: {plot_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output log file: {output_log_file}")

    # Update output filenames
    plot_filename = f"{output_basename}_{feature_type}{pca_tag}_tica_lag{lag_time}_dim{n_dim}_plot.png"
    tica_model_filename = f"{output_basename}_{feature_type}{pca_tag}_tica_lag{lag_time}_dim{n_dim}_model.joblib"
    features_data_filename = f"{output_basename}_features_{feature_type}.npy"
    pca_model_filename = f"{output_basename}_features_{feature_type}_pca_model.joblib"
    pca_features_data_filename = f"{output_basename}_features_{feature_type}{pca_tag}_pca_transformed.npy"
    tica_output_filename = f"{output_basename}_{feature_type}{pca_tag}_tica_output_lag{lag_time}_dim{n_dim}.npy"

    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_model_file = os.path.join(output_dir, tica_model_filename)
    output_features_data_file = os.path.join(output_dir, features_data_filename)
    output_pca_model_file = os.path.join(output_dir, pca_model_filename)
    output_pca_features_data_file = os.path.join(output_dir, pca_features_data_filename)
    output_tica_output_file = os.path.join(output_dir, tica_output_filename)

    logger.info(f"Output Raw Features file: {output_features_data_file}")
    if pca_variance_cutoff > 0:
        logger.info(f"Output PCA Model file: {output_pca_model_file}")
        logger.info(f"Output PCA Features file: {output_pca_features_data_file}")
    logger.info(f"Output TICA Model file: {output_tica_model_file}")
    logger.info(f"Output TICA Projection file: {output_tica_output_file}")
    logger.info(f"Output Plot file: {output_plot_file}")


    pca_model = None
    tica_model = None
    tica_input_data = None # This will hold either raw features or PCA features

    if not os.path.exists(pdb_file):
        logger.error(f"Error: Input PDB file not found at '{pdb_file}'")
        sys.exit(1)

    try:
        logger.info("Loading trajectory...")
        try:
            traj = md.load(pdb_file)
            if traj is None or traj.n_frames == 0 or traj.n_atoms == 0:
                msg = f"Failed to load a valid trajectory from {pdb_file}. Check PDB format and content."
                logger.error(msg)
                raise ValueError(msg)
            topology_mdtraj = traj.topology
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
            if traj.n_frames == lag_time:
                logger.warning(f"TICA calculation might fail or be unstable as n_frames == lag_time.")

        # 1. Calculate Raw Features
        logger.info(f"Calculating raw features ({feature_type})...")
        raw_features_data, needs_alignment = calculate_features(traj, feature_type, logger)
        logger.info(f"Raw feature calculation complete. Shape: {raw_features_data.shape}")

        # Handle alignment if needed (applies to trajectory coords, affects heavy_atom features)
        if needs_alignment:
            if traj.n_frames < 2:
                 logger.warning("Warning: Only 1 frame, cannot perform alignment (needed for heavy_atom features).")
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
                            raw_features_data, _ = calculate_features(traj, feature_type, logger)
                            logger.info(f"Recalculated raw feature shape: {raw_features_data.shape}")
                    except Exception as e:
                        logger.error(f"Error during trajectory alignment: {e}. Proceeding with unaligned coordinates for heavy atoms, results may be affected.")

        # Check for NaNs/Infs in raw features
        if not np.all(np.isfinite(raw_features_data)):
             num_nonfinite = np.sum(~np.isfinite(raw_features_data))
             msg = f"Raw feature data contains {num_nonfinite} non-finite values (NaN or Inf). Cannot proceed. Check feature calculation ('{feature_type}')."
             logger.error(msg)
             raise ValueError(msg)

        # Save raw features
        try:
            logger.info(f"Saving raw features data to '{output_features_data_file}'...")
            np.save(output_features_data_file, raw_features_data)
            logger.info("Raw features data saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save raw features data to '{output_features_data_file}'. Continuing. Details: {e}")


        # 2. Optional PCA Pre-processing
        if pca_variance_cutoff > 0:
            logger.info(f"Applying PCA with variance cutoff {pca_variance_cutoff:.2f}...")
            # Ensure cutoff is within valid range
            if not (0 < pca_variance_cutoff <= 1.0):
                 logger.warning(f"PCA variance cutoff {pca_variance_cutoff} is outside (0, 1]. Disabling PCA.")
                 tica_input_data = raw_features_data
            else:
                try:
                    pca_estimator = PCA(n_components=pca_variance_cutoff, svd_solver='auto')
                    logger.info("Fitting PCA model...")
                    pca_model = pca_estimator.fit(raw_features_data)
                    n_components_pca = pca_model.n_components_
                    explained_variance_ratio = np.sum(pca_model.explained_variance_ratio_)
                    logger.info(f"PCA fitted. Kept {n_components_pca} components, explaining {explained_variance_ratio:.4f} of the variance.")

                    logger.info("Transforming features using PCA...")
                    tica_input_data = pca_model.transform(raw_features_data)
                    logger.info(f"PCA transformation complete. PCA-transformed feature shape: {tica_input_data.shape}")

                    # Save PCA model
                    try:
                        logger.info(f"Saving PCA model to '{output_pca_model_file}'...")
                        joblib.dump(pca_model, output_pca_model_file)
                        logger.info("PCA model saved successfully.")
                    except Exception as e:
                        logger.warning(f"Warning: Failed to save PCA model to '{output_pca_model_file}'. Continuing. Details: {e}")

                    # Save PCA-transformed features
                    try:
                        logger.info(f"Saving PCA-transformed features data to '{output_pca_features_data_file}'...")
                        np.save(output_pca_features_data_file, tica_input_data)
                        logger.info("PCA-transformed features data saved successfully.")
                    except Exception as e:
                         logger.warning(f"Warning: Failed to save PCA-transformed features data to '{output_pca_features_data_file}'. Continuing. Details: {e}")

                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.error(f"Error during PCA fitting/transformation: {e}")
                    logger.error("This might happen with insufficient data variability or invalid cutoff. Check data and parameters. Aborting.")
                    sys.exit(1)
        else:
            logger.info("Skipping PCA pre-processing.")
            tica_input_data = raw_features_data

        # --- Log summary statistics of the TICA input features ---
        if tica_input_data.size > 0:
            data_source = "PCA-transformed" if pca_variance_cutoff > 0 else "Raw"
            logger.info(f"{data_source} feature data summary (input to TICA, first 5 dim if >5):")
            num_features_to_log = min(5, tica_input_data.shape[1])
            logger.info(f"  Shape: {tica_input_data.shape}")
            try:
                mean_vals = np.mean(tica_input_data[:, :num_features_to_log], axis=0)
                std_vals = np.std(tica_input_data[:, :num_features_to_log], axis=0)
                min_vals = np.min(tica_input_data[:, :num_features_to_log], axis=0)
                max_vals = np.max(tica_input_data[:, :num_features_to_log], axis=0)
                logger.info(f"  Mean (first {num_features_to_log}): {mean_vals}")
                logger.info(f"  Std Dev (first {num_features_to_log}): {std_vals}")
                logger.info(f"  Min (first {num_features_to_log}): {min_vals}")
                logger.info(f"  Max (first {num_features_to_log}): {max_vals}")
            except Exception as stat_err:
                logger.warning(f"Could not compute summary statistics for {data_source} features: {stat_err}")
        else:
            logger.warning(f"{data_source} feature data is empty, cannot compute summary statistics.")
        # --- End logging summary statistics ---


        # 3. Perform TICA on the (potentially PCA-reduced) data
        logger.info(f"Performing TICA calculation with lag={lag_time} using deeptime...")
        actual_n_frames = tica_input_data.shape[0]
        tica_input_dim = tica_input_data.shape[1]

        # Check dimension feasibility *after* potential PCA
        if actual_n_frames <= lag_time:
             msg = f"Cannot compute TICA: Data length ({actual_n_frames}) <= lag time ({lag_time}) after potential PCA."
             logger.error(msg)
             raise ValueError(msg)

        max_tica_dim = min(tica_input_dim, actual_n_frames - 1) # Theoretical max based on input shape
        if max_tica_dim < 1:
             msg = f"Input dimension to TICA is {tica_input_dim} with {actual_n_frames} frames. Cannot compute TICA (max possible dim < 1)."
             logger.error(msg)
             raise ValueError(msg)

        if n_dim > max_tica_dim:
            logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) exceeds maximum possible ({max_tica_dim}) based on TICA input data. Reducing to {max_tica_dim}.")
            n_dim = max_tica_dim
        elif n_dim <= 0:
             logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) must be >= 1. Setting to 1.")
             n_dim = 1

        # Instantiate and fit deeptime TICA estimator
        tica_estimator = TICA(lagtime=lag_time, dim=n_dim, scaling='kinetic_map')

        # Fit the TICA model
        try:
            logger.info(f"Fitting TICA model on data with shape {tica_input_data.shape}...")
            tica_estimator.fit(tica_input_data)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"Error during TICA fitting (lag={lag_time}, dim={n_dim}): {e}")
            logger.error("This might be due to insufficient data variability for the given lag time, rank deficiency in the TICA input data, or other numerical issues.")
            sys.exit(1)

        # Fetch the trained TICA model
        tica_model = tica_estimator.fetch_model()
        if tica_model is None:
            logger.error("Failed to fetch TICA model after fitting.")
            sys.exit(1)

        # Transform the data (get the TICA projection)
        logger.info("Transforming features using TICA model...")
        tica_output = tica_model.transform(tica_input_data)

        actual_dims_computed = tica_output.shape[1]
        if actual_dims_computed == 0:
             logger.error(f"TICA computation resulted in 0 dimensions despite requesting {n_dim}. Aborting.")
             sys.exit(1)
        if actual_dims_computed < n_dim:
             logger.warning(f"TICA computation resulted in {actual_dims_computed} dimensions, which is less than the requested {n_dim}. This might indicate rank deficiency or intrinsic data properties.")
             # Use actual_dims_computed for plotting etc.
             n_dim_plot = actual_dims_computed
        else:
            n_dim_plot = n_dim

        logger.info(f"TICA calculation complete. Output projection shape: {tica_output.shape}")
        try:
            computed_timescales = tica_model.timescales(lagtime=lag_time)
            logger.info(f"TICA Timescales (lag={lag_time} K): {computed_timescales[:n_dim_plot]}") # Show only relevant timescales
        except Exception as e:
            logger.warning(f"Could not compute timescales: {e}")

        # Save TICA projection
        try:
            logger.info(f"Saving TICA projection data to '{output_tica_output_file}'...")
            np.save(output_tica_output_file, tica_output)
            logger.info("TICA projection data saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save TICA projection data to '{output_tica_output_file}'. Continuing. Details: {e}")

        # Save TICA model
        try:
            logger.info(f"Saving TICA model object to '{output_tica_model_file}'...")
            joblib.dump(tica_model, output_tica_model_file)
            logger.info("TICA model object saved successfully.")
        except Exception as e:
            logger.error(f"Error: Failed to save TICA model to '{output_tica_model_file}'. Cannot continue. Details: {e}")
            sys.exit(1)


        # 4. Generate Plot based on final TICA projection
        logger.info(f"Generating TICA plot...")
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dim = min(2, n_dim_plot) # Use actual computed dimensions for plotting
        pca_title_part = f"PCA {int(pca_variance_cutoff*100)}% -> {tica_input_dim}D, " if pca_variance_cutoff > 0 else ""
        plot_title_base = f'TICA Projection ({pca_title_part}Lag: {lag_time}, Feat: {feature_type})'

        if plot_dim == 0:
            logger.warning("Warning: TICA resulted in 0 dimensions. Cannot generate plot.")
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

        elif plot_dim >= 2:
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
                    if np.isclose(np.var(x), 0) or np.isclose(np.var(y), 0):
                        logger.warning("Warning: Very low variance in TICA components. Hist2d may fail or be uninformative. Falling back to scatter plot.")
                        scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label('Frame Index')
                        plot_title += " (hist2d failed due to low variance, showing scatter)"
                    else:
                        counts, xedges, yedges, image = ax.hist2d(
                            x, y, bins=100, cmap='viridis', cmin=1, norm=LogNorm() # Added LogNorm often useful
                        )
                        cbar = fig.colorbar(image, ax=ax)
                        cbar.set_label('Counts (log scale)')
                except ValueError as e:
                    logger.warning(f"Warning: hist2d plot failed: {e}. Check data range/variance. Falling back to scatter plot.")
                    scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Frame Index')
                    plot_title += " (hist2d failed, showing scatter)"
            else:
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
        pass # Error already logged
    except MemoryError:
        logger.error(f"Error: Insufficient memory during analysis. Consider reducing PCA variance cutoff, using fewer features, or downsampling data.")
        sys.exit(1)
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
         logger.error(f"Error during PCA/TICA analysis setup or calculation: {e}")
         import traceback
         logger.debug(traceback.format_exc())
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logging.shutdown()

    print("--- PCA-TICA Analysis Finished ---")
    return pca_model, tica_model # Return both models

def main():
    parser = argparse.ArgumentParser(
        description='Perform optional PCA and then TICA on a PDB trajectory using scikit-learn and deeptime.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pdb_file',
        help='Path to the input multi-frame PDB file (trajectory).'
    )
    parser.add_argument(
        '--features', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'],
        default='backbone_torsions',
        help='Type of features to use for analysis.'
    )
    # PCA Arguments
    parser.add_argument(
        '--pca-variance', type=float, default=0.95,
        help='Fraction of variance to keep with PCA pre-processing. Set to 0 or <= 0 to disable PCA.'
    )
    # TICA Arguments
    parser.add_argument(
        '-l', '--lag', type=int, default=10,
        help='TICA lag time (in frames). Must be > 0.'
    )
    parser.add_argument(
        '-d', '--dim', type=int, default=2,
        help='Number of TICA dimensions to calculate (>= 1) from the (potentially PCA-reduced) features.'
    )
    # Output Arguments
    parser.add_argument(
        '-o', '--output-dir', default="pca_tica_analysis_output",
        help='Directory to save plots, models, intermediate files, and log.'
    )
    parser.add_argument(
        '--plot-type', choices=['scatter', 'hist2d'], default='scatter',
        help='Type of plot for TICA projection (first 2 components).'
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
    if args.pca_variance > 1.0 :
        print(f"Warning: PCA variance cutoff {args.pca_variance} > 1.0. Setting to 1.0 (keep all components).", file=sys.stderr)
        args.pca_variance = 1.0
        # Note: pca_variance <= 0 explicitly disables it in run_tica_analysis
    if args.lag <= 0:
        print("Error: TICA Lag time (--lag) must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.dim < 1:
        print("Error: TICA dimensions (--dim) must be >= 1.", file=sys.stderr)
        sys.exit(1)
    # --- End Input Validation ---

    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(args.pdb_file))[0]

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
        plot_type=args.plot_type,
        pca_variance_cutoff=args.pca_variance # Pass the new argument
    )

if __name__ == "__main__":
    main()
