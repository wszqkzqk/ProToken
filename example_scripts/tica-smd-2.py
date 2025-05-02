#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyemma')

try:
    import mdtraj as md
    import pyemma
    import pyemma.coordinates as coor
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy, argparse.")
    logging.error(f"Import error details: {e}")
    sys.exit(1)

def setup_logging(log_file):
    """Configure logger."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear previous handlers if any exist (important for re-runs in interactive sessions)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    return root_logger

def calculate_custom_backbone_torsions(traj, logger, output_dir, output_basename):
    """
    Calculates sin/cos of backbone dihedral angles (phi, psi, omega) using MDTraj.
    Saves raw angles to an .npz file.
    Returns a numpy array (n_frames, n_features).
    """
    logger.info("Calculating features using CUSTOM backbone torsions (sin/cos via MDTraj)...")
    try:
        phi_indices, phi_angles = md.compute_phi(traj)
        psi_indices, psi_angles = md.compute_psi(traj)
        omega_indices, omega_angles = md.compute_omega(traj)
        logger.info(f"MDTraj computed angles per frame: {phi_angles.shape[1]} phi, {psi_angles.shape[1]} psi, {omega_angles.shape[1]} omega")

        # --- Save Raw Angles ---
        raw_angles_filename = os.path.join(output_dir, f"{output_basename}_raw_backbone_torsions.npz")
        try:
            logger.info(f"Saving raw backbone torsion angles (phi, psi, omega) to '{raw_angles_filename}'...")
            np.savez(raw_angles_filename, phi=phi_angles, psi=psi_angles, omega=omega_angles)
            logger.info("Raw angles saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save raw angles to '{raw_angles_filename}'. Continuing analysis. Details: {e}")
        # --- End Save Raw Angles ---

        # Check for empty results (e.g., single residue)
        if phi_angles.size == 0 and psi_angles.size == 0 and omega_angles.size == 0:
            raise ValueError("No backbone torsion angles could be computed by MDTraj.")

        # Calculate sin/cos and concatenate
        # Order: sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
        components = []
        if phi_angles.size > 0:
            components.extend([np.sin(phi_angles), np.cos(phi_angles)])
        if psi_angles.size > 0:
            components.extend([np.sin(psi_angles), np.cos(psi_angles)])
        if omega_angles.size > 0:
            # Omega can fluctuate around pi/-pi. Using sin/cos handles periodicity naturally.
            components.extend([np.sin(omega_angles), np.cos(omega_angles)])

        if not components:
             raise ValueError("No valid torsion components to concatenate.")

        features_data = np.concatenate(components, axis=1).astype(np.float32) # Use float32 like PyEMMA often does
        logger.info(f"Custom backbone torsion feature calculation complete. Feature shape: {features_data.shape}")
        return features_data

    except Exception as e:
        logger.error(f"Error during custom backbone torsion calculation: {e}")
        # Re-raise as RuntimeError to be caught by the main try-except block
        raise RuntimeError("Failed to compute custom features.") from e

def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type, plot_type):
    """
    Performs TICA analysis... (docstring unchanged)
    """
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}.log"
    output_log_file = os.path.join(output_dir, log_filename)
    logger = setup_logging(output_log_file)

    logger.info("--- Starting TICA Analysis ---")
    logger.info(f"Input PDB: {pdb_file}")
    logger.info(f"Lag time: {lag_time} frames")
    logger.info(f"Target TICA dimensions: {n_dim}")
    logger.info(f"Feature type: {feature_type}")
    logger.info(f"Plot type: {plot_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output log file: {output_log_file}")

    # --- Adjusted filename for clarity if using custom features ---
    if feature_type == 'backbone_torsions':
        feature_desc_for_filename = 'custom_backbone_torsions'
    else:
        feature_desc_for_filename = feature_type

    plot_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_desc_for_filename}_plot.png"
    tica_obj_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_desc_for_filename}_model.pyemma"
    # No featurizer to save for custom features
    featurizer_filename = f"{output_basename}_featurizer_{feature_desc_for_filename}.pyemma" # Keep name consistent for non-custom
    features_data_filename = f"{output_basename}_features_{feature_desc_for_filename}.npy"
    tica_output_filename = f"{output_basename}_tica_output_lag{lag_time}_dim{n_dim}_{feature_desc_for_filename}.npy"
    # Add filename for raw angles if using custom torsions
    raw_angles_output_file = None
    if feature_type == 'backbone_torsions':
        raw_angles_output_file = os.path.join(output_dir, f"{output_basename}_raw_backbone_torsions.npz")


    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_file = os.path.join(output_dir, tica_obj_filename)
    output_featurizer_file = os.path.join(output_dir, featurizer_filename)
    output_features_data_file = os.path.join(output_dir, features_data_filename)
    output_tica_output_file = os.path.join(output_dir, tica_output_filename)

    logger.info(f"Output plot file: {output_plot_file}")
    logger.info(f"Output TICA object file: {output_tica_file}")
    logger.info(f"Output Featurizer file: {output_featurizer_file if feature_type != 'backbone_torsions' else 'N/A (custom features)'}")
    logger.info(f"Output Features data file: {output_features_data_file}")
    logger.info(f"Output TICA projection file: {output_tica_output_file}")
    if raw_angles_output_file:
        logger.info(f"Output Raw Angles file: {raw_angles_output_file}") # Log the raw angles file path

    tica_obj = None

    if not os.path.exists(pdb_file):
        logger.error(f"Error: Input PDB file not found at '{pdb_file}'")
        sys.exit(1)

    try:
        logger.info("Loading topology from the first frame...")
        try:
            # Load topology separately first for featurizer setup (if needed)
            topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology
        except Exception as e:
             logger.error(f"Error: Failed to load topology from {pdb_file}. Check format. Details: {e}")
             sys.exit(1)

        if topology_mdtraj is None or topology_mdtraj.n_atoms == 0:
             logger.error("Could not load a valid topology from the PDB file.")
             raise ValueError("Could not load a valid topology from the PDB file.")
        logger.info(f"Topology loaded: {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")

        logger.info("Loading full trajectory...")
        # Use topology when loading the full trajectory for consistency
        traj = md.load(pdb_file, top=topology_mdtraj)
        logger.info(f"Loaded trajectory with {traj.n_frames} frames.")

        # --- Basic Trajectory Checks ---
        if traj.n_frames == 0:
             msg = "Input PDB file contains 0 frames."
             logger.error(msg)
             raise ValueError(msg)
        if traj.n_frames <= lag_time:
            logger.warning(f"Warning: Trajectory length ({traj.n_frames}) <= lag time ({lag_time}). TICA results might be unreliable or fail.")
            if traj.n_frames <= 1:
                 msg = "Cannot compute TICA with <= 1 frame."
                 logger.error(msg)
                 raise ValueError(msg)

        # --- Alignment (Only for coordinate-based features like heavy_atom) ---
        needs_alignment = (feature_type == 'heavy_atom')
        if needs_alignment:
            if traj.n_frames < 2:
                 logger.warning("Warning: Only 1 frame in trajectory, cannot perform alignment.")
            else:
                 logger.info("Aligning trajectory to the first frame (using heavy atoms)...")
                 try:
                    heavy_atom_indices = topology_mdtraj.select('not element H and protein')
                    if len(heavy_atom_indices) == 0:
                        raise ValueError("No heavy atoms found for alignment.")
                    traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)
                 except Exception as align_err:
                    logger.error(f"Alignment failed: {align_err}")
                    raise RuntimeError("Alignment required but failed.") from align_err


        # --- Feature Calculation ---
        features_data = None
        feat = None # Initialize featurizer object

        # <<< MODIFIED FEATURE LOGIC >>>
        if feature_type == 'backbone_torsions':
            # Use the new custom function, passing output dir and basename
            features_data = calculate_custom_backbone_torsions(traj, logger, output_dir, output_basename)
            # No PyEMMA featurizer object is created or saved in this case

        elif feature_type == 'ca_dist':
            logger.info("Setting up PyEMMA featurizer for C-alpha distances...")
            feat = coor.featurizer(topology_mdtraj)
            ca_indices = topology_mdtraj.select('name CA and protein')
            if len(ca_indices) < 2:
                msg = f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found)."
                logger.error(msg)
                raise ValueError(msg)
            feat.add_distances_ca()
            logger.info(f"Using PyEMMA C-alpha distances. Feature dimension: {feat.dimension()}")
            logger.info("Calculating features using PyEMMA featurizer...")
            features_data = feat.transform(traj)

        elif feature_type == 'heavy_atom':
            logger.info("Setting up PyEMMA featurizer for heavy atom coordinates...")
            feat = coor.featurizer(topology_mdtraj)
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if len(heavy_atom_indices) == 0:
                 msg = "No heavy atoms found in protein residues."
                 logger.error(msg)
                 raise ValueError(msg)
            feat.add_selection(heavy_atom_indices)
            # Alignment was done earlier if needed
            logger.info(f"Using PyEMMA heavy atom coordinates. Feature dimension: {feat.dimension()}")
            logger.info("Calculating features using PyEMMA featurizer...")
            features_data = feat.transform(traj)

        else:
             msg = f"Invalid feature type '{feature_type}' specified."
             logger.error(msg)
             raise ValueError(msg)

        # Save PyEMMA featurizer *if* it was created
        if feat is not None:
            try:
                logger.info(f"Saving PyEMMA featurizer object to '{output_featurizer_file}'...")
                feat.save(output_featurizer_file, overwrite=True)
                logger.info("Featurizer object saved successfully.")
            except Exception as e:
                logger.warning(f"Warning: Failed to save featurizer object to '{output_featurizer_file}'. Continuing analysis. Details: {e}")
        # <<< END OF MODIFIED FEATURE LOGIC >>>


        # --- Checks After Feature Calculation ---
        if features_data is None or features_data.shape[0] == 0:
             msg = "Feature calculation resulted in empty data."
             logger.error(msg)
             raise RuntimeError(msg)
        if features_data.shape[0] != traj.n_frames:
             # This check is important, especially with custom features
             msg = f"Mismatch in number of frames between trajectory ({traj.n_frames}) and features ({features_data.shape[0]})"
             logger.error(msg)
             raise RuntimeError(msg)
        logger.info(f"Feature calculation complete. Feature shape: {features_data.shape}")


        # --- Log summary statistics of the features (unchanged) ---
        log_features = features_data # No list handling needed now
        if log_features.size > 0:
            logger.info(f"Feature data summary (first 5 features if >5):")
            num_features_to_log = min(5, log_features.shape[1])
            logger.info(f"  Shape: {log_features.shape}")
            try:
                # Ensure calculations are done on valid data
                valid_features = log_features[:, :num_features_to_log]
                if np.isnan(valid_features).any() or np.isinf(valid_features).any():
                     logger.warning("NaN or Inf detected in features, statistics might be unreliable.")
                mean_vals = np.nanmean(valid_features, axis=0)
                std_vals = np.nanstd(valid_features, axis=0)
                min_vals = np.nanmin(valid_features, axis=0)
                max_vals = np.nanmax(valid_features, axis=0)
                logger.info(f"  Mean (first {num_features_to_log}): {mean_vals}")
                logger.info(f"  Std Dev (first {num_features_to_log}): {std_vals}")
                logger.info(f"  Min (first {num_features_to_log}): {min_vals}")
                logger.info(f"  Max (first {num_features_to_log}): {max_vals}")
            except Exception as stat_err:
                logger.warning(f"Could not compute summary statistics for features: {stat_err}")
        else:
            logger.warning("Feature data is empty, cannot compute summary statistics.")


        # --- Save Features Data (unchanged) ---
        try:
            logger.info(f"Saving features data to '{output_features_data_file}'...")
            np.save(output_features_data_file, features_data)
            logger.info("Features data saved successfully.")
            logger.info(f"To inspect the full features data, load with: `import numpy as np; features = np.load('{output_features_data_file}')`")
        except Exception as e:
            logger.warning(f"Warning: Failed to save features data to '{output_features_data_file}'. Continuing analysis. Details: {e}")


        # --- Perform TICA Calculation (Input is now unified features_data) ---
        logger.info(f"Performing TICA calculation with lag={lag_time}...")

        actual_n_frames = features_data.shape[0]
        feature_dim = features_data.shape[1]

        # --- Dimension checks for TICA (Ensure sufficient data) ---
        if actual_n_frames <= lag_time + 1:
             # Updated check: Need at least lag+2 frames to compute covariance estimators
             msg = f"Cannot compute TICA. Insufficient effective samples. Frames={actual_n_frames}, lag={lag_time}. Need > lag+1 frames."
             logger.error(msg)
             raise ValueError(msg)

        # Maximum possible rank for TICA depends on covariance matrices
        # A safe upper bound is min(feature_dim, actual_n_frames - lag_time - 1) but implementation details might differ slightly.
        # Let PyEMMA handle internal rank determination based on epsilon cutoff, but check n_dim against feature_dim.
        if feature_dim == 0:
             msg = "Cannot compute TICA. Feature dimension is 0."
             logger.error(msg)
             raise ValueError(msg)

        max_allowable_dim = feature_dim # Theoretical max based on features
        if n_dim > max_allowable_dim:
             logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) exceeds feature dimensions ({max_allowable_dim}). Reducing to {max_allowable_dim}.")
             n_dim = max_allowable_dim


        # --- Call PyEMMA TICA ---
        # Ensure data is float32 or float64, not integers if custom features happened to be that way
        if not np.issubdtype(features_data.dtype, np.floating):
            logger.warning(f"Feature data type is {features_data.dtype}, converting to float64 for TICA.")
            features_data = features_data.astype(np.float64)

        tica_obj = coor.tica(features_data, lag=lag_time, dim=n_dim, kinetic_map=True, reversible=False) # Default reversible=True
        # Consider adding ", reversible=False" here if default still gives bad results

        tica_output = tica_obj.get_output()[0] # Assuming single trajectory input
        # <<< Check output shape again rigorously >>>
        if tica_output.shape[0] != actual_n_frames:
             logger.error(f"TICA output frame count ({tica_output.shape[0]}) does not match input feature frame count ({actual_n_frames})!")
             # This shouldn't happen with standard PyEMMA, but good to check.
             raise RuntimeError("TICA output frame count mismatch.")
        if tica_output.shape[1] == 0 and n_dim > 0:
             logger.warning(f"TICA computation resulted in 0 output dimensions despite requesting {n_dim}. Check eigenvalues/variance.")
        # <<< End Check >>>

        logger.info(f"TICA calculation complete. Output projection shape: {tica_output.shape}")
        logger.info(f"TICA Timescales: {tica_obj.timescales}")
        logger.info(f"TICA Eigenvalues: {tica_obj.eigenvalues}")


        # --- Save TICA Output and Object (unchanged) ---
        try:
            logger.info(f"Saving TICA projection data to '{output_tica_output_file}'...")
            np.save(output_tica_output_file, tica_output)
            logger.info("TICA projection data saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save TICA projection data to '{output_tica_output_file}'. Continuing analysis. Details: {e}")

        try:
            logger.info(f"Saving TICA object (model) to '{output_tica_file}'...")
            tica_obj.save(output_tica_file, overwrite=True)
            logger.info("TICA object saved successfully.")
        except Exception as e:
            # Saving the TICA object might fail if inputs were weird
            logger.error(f"Error: Failed to save TICA object to '{output_tica_file}'. Cannot continue. Details: {e}")
            sys.exit(1)


        # --- Generate TICA Plot (largely unchanged) ---
        logger.info(f"Generating TICA plot...")
        actual_dims_computed = tica_output.shape[1]
        if actual_dims_computed == 0:
             logger.warning("Warning: TICA computation resulted in 0 dimensions. Cannot generate plot.")
        else:
            # Determine plot title based on features used
            if feature_type == 'backbone_torsions':
                feature_label_for_plot = 'Backbone Torsions'
            else:
                feature_label_for_plot = f'PyEMMA {feature_type}'

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_dim = min(2, actual_dims_computed)

            if plot_dim == 2:
                x = tica_output[:, 0]
                y = tica_output[:, 1]
                xlabel = 'TICA Component 1 (IC1)'
                ylabel = 'TICA Component 2 (IC2)'
                plot_title = f'TICA Projection (Lag: {lag_time} frames, Features: {feature_label_for_plot})'

                # --- Plotting logic (scatter / hist2d) - unchanged ---
                if plot_type == 'scatter':
                    scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Frame Index')
                elif plot_type == 'hist2d':
                    # ... (hist2d logic unchanged) ...
                    try:
                        from matplotlib.colors import LogNorm
                        counts, xedges, yedges, image = ax.hist2d(
                            x, y, bins=100, cmap='viridis', cmin=1 # , norm=LogNorm() # Consider LogNorm for better visibility
                        )
                        cbar = fig.colorbar(image, ax=ax)
                        cbar.set_label('Counts')
                    except ValueError as e:
                       logger.warning(f"Warning: hist2d plot failed: {e}. Check data range/variance. Falling back to scatter.")
                       scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                       cbar = fig.colorbar(scatter, ax=ax)
                       cbar.set_label('Frame Index')
                       plot_title += " (hist2d failed, showing scatter)"

                else:
                    msg = f"Invalid plot_type: {plot_type}"
                    logger.error(msg)
                    raise ValueError(msg)

            elif plot_dim == 1:
                # ... (1D plot logic unchanged) ...
                logger.info("Plotting TICA Component 1 vs Frame Index (Only 1 dimension computed).")
                x = np.arange(len(tica_output))
                y = tica_output[:, 0]
                xlabel = 'Frame Index'
                ylabel = 'TICA Component 1'
                plot_title = f'TICA Component 1 (Lag: {lag_time} frames, Features: {feature_label_for_plot})'
                ax.plot(x, y, marker='.', linestyle='-', markersize=2) # Use plot instead of scatter for 1D time series

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(output_plot_file, dpi=300)
            logger.info(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig)

    # --- Error Handling and Finalization (unchanged) ---
    except FileNotFoundError:
        # This case is handled earlier now, but kept for safety
        logger.error(f"Error: Input PDB file not found at '{pdb_file}'") # Logged earlier
        sys.exit(1)
    except MemoryError:
        logger.error(f"Error: Insufficient memory during analysis. Consider using smaller features or reducing trajectory size.")
        sys.exit(1)
    except ValueError as e:
         logger.error(f"Error during TICA analysis setup or calculation: {e}")
         sys.exit(1)
    except RuntimeError as e: # Catch RuntimeErrors from custom featurizer or alignment
         logger.error(f"Runtime error during analysis: {e}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during TICA analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Ensure logging handlers are removed to prevent duplicate messages on re-runs
        if 'logger' in locals() and isinstance(logger, logging.Logger):
             handlers = logger.handlers[:]
             for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        # logging.shutdown() # Sometimes causes issues if run multiple times; manual removal is safer

    print("--- TICA Analysis Finished ---")
    return tica_obj # May be None if analysis failed early


def main():
    parser = argparse.ArgumentParser(
        description='Perform TICA on a PDB trajectory, save the TICA object, projection plot, intermediate files, and logs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pdb_file',
        help='Path to the input multi-frame PDB file (trajectory).'
    )
    parser.add_argument(
        '-l', '--lag', type=int, default=10,
        help='TICA lag time (in frames).'
    )
    parser.add_argument(
        '-d', '--dim', type=int, default=2,
        help='Number of TICA dimensions to calculate (>= 1).'
    )
    parser.add_argument(
        '-o', '--output-dir', default="tica_analysis_output",
        help='Directory to save the TICA projection plot, TICA object file, intermediate files, and log.'
    )
    parser.add_argument(
        '--features', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'], # Keep backbone_torsions as the trigger
        default='backbone_torsions',
        help='Type of features to use. "backbone_torsions" now uses custom MDTraj implementation.'
    )
    parser.add_argument(
        '--plot-type', choices=['scatter', 'hist2d'], default='scatter',
        help='Type of plot for TICA projection. "scatter" is better for paths/sparse data.'
    )
    parser.add_argument(
        '--basename', default=None,
        help='Basename for output files. Defaults to input PDB filename without extension.'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.lag <= 0:
        print("Error: Lag time (--lag) must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.dim < 1:
        print("Error: TICA dimensions (--dim) must be >= 1.", file=sys.stderr)
        sys.exit(1)

    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(args.pdb_file))[0]

    # Call the analysis function
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
