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

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    return root_logger

def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type, plot_type):
    """
    Performs TICA analysis on a multi-frame PDB file using PyEMMA,
    saves the TICA object, projection plot, intermediate files, and logs
    to the specified directory.

    Parameters
    ----------
    pdb_file : str
        Path to the input multi-frame PDB file.
    lag_time : int
        TICA lag time in units of frames.
    n_dim : int
        Number of TICA dimensions to compute.
    output_dir : str
        Directory where the plot, TICA object, intermediate files, and log will be saved.
    output_basename : str
        Base name for the output files.
    feature_type : str
        Type of features to use ('ca_dist', 'heavy_atom', 'backbone_torsions').
    plot_type : str
        Type of plot ('scatter' or 'hist2d').

    Returns
    -------
    tica_obj : pyemma.coordinates.TICA or None
        The computed TICA object, or None if an error occurred before computation.
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

    plot_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}_plot.png"
    tica_obj_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}_model.pyemma"
    featurizer_filename = f"{output_basename}_featurizer_{feature_type}.pyemma"
    features_data_filename = f"{output_basename}_features_{feature_type}.npy"
    tica_output_filename = f"{output_basename}_tica_output_lag{lag_time}_dim{n_dim}_{feature_type}.npy"

    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_file = os.path.join(output_dir, tica_obj_filename)
    output_featurizer_file = os.path.join(output_dir, featurizer_filename)
    output_features_data_file = os.path.join(output_dir, features_data_filename)
    output_tica_output_file = os.path.join(output_dir, tica_output_filename)

    logger.info(f"Output plot file: {output_plot_file}")
    logger.info(f"Output TICA object file: {output_tica_file}")
    logger.info(f"Output Featurizer file: {output_featurizer_file}")
    logger.info(f"Output Features data file: {output_features_data_file}")
    logger.info(f"Output TICA projection file: {output_tica_output_file}")

    tica_obj = None

    if not os.path.exists(pdb_file):
        logger.error(f"Error: Input PDB file not found at '{pdb_file}'")
        sys.exit(1)

    try:
        logger.info("Loading topology from the first frame...")
        try:
            topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology
        except Exception as e:
             logger.error(f"Error: Failed to load topology from {pdb_file}. Check format. Details: {e}")
             sys.exit(1)

        if topology_mdtraj is None or topology_mdtraj.n_atoms == 0:
             logger.error("Could not load a valid topology from the PDB file.")
             raise ValueError("Could not load a valid topology from the PDB file.")
        logger.info(f"Topology loaded: {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")

        logger.info("Setting up featurizer...")
        feat = coor.featurizer(topology_mdtraj)
        features_data = None
        needs_alignment = False

        if feature_type == 'ca_dist':
            ca_indices = topology_mdtraj.select('name CA and protein')
            if len(ca_indices) < 2:
                msg = f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found)."
                logger.error(msg)
                raise ValueError(msg)
            feat.add_distances_ca()
            logger.info(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")

        elif feature_type == 'heavy_atom':
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if len(heavy_atom_indices) == 0:
                 msg = "No heavy atoms found in protein residues."
                 logger.error(msg)
                 raise ValueError(msg)
            feat.add_selection(heavy_atom_indices)
            needs_alignment = True
            logger.info(f"Using heavy atom coordinates. Feature dimension: {feat.dimension()}")

        elif feature_type == 'backbone_torsions':
            feat.add_backbone_torsions(cossin=True, periodic=False)
            logger.info(f"Using backbone torsions (cos/sin). Feature dimension: {feat.dimension()}")
            if feat.dimension() == 0:
                msg = "Could not find any backbone torsions. Is it a valid protein structure?"
                logger.error(msg)
                raise ValueError(msg)

        else:
             msg = f"Invalid feature type '{feature_type}' specified."
             logger.error(msg)
             raise ValueError(msg)

        try:
            logger.info(f"Saving featurizer object to '{output_featurizer_file}'...")
            feat.save(output_featurizer_file, overwrite=True)
            logger.info("Featurizer object saved successfully.")
        except Exception as e:
            logger.warning(f"Warning: Failed to save featurizer object to '{output_featurizer_file}'. Continuing analysis. Details: {e}")

        logger.info("Loading full trajectory...")
        traj = md.load(pdb_file, top=topology_mdtraj)
        logger.info(f"Loaded trajectory with {traj.n_frames} frames.")

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

        if needs_alignment:
            if traj.n_frames < 2:
                 logger.warning("Warning: Only 1 frame in trajectory, cannot perform alignment.")
            else:
                 logger.info("Aligning trajectory to the first frame (using heavy atoms)...")
                 heavy_atom_indices = topology_mdtraj.select('not element H and protein')
                 traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)

        logger.info(f"Calculating features ({feature_type})...")
        features_data = feat.transform(traj)

        if features_data is None or features_data.shape[0] == 0:
             msg = "Feature calculation resulted in empty data."
             logger.error(msg)
             raise RuntimeError(msg)
        logger.info(f"Feature calculation complete. Feature shape: {features_data.shape}")

        # --- Log summary statistics of the features ---
        if isinstance(features_data, list): # Handle list output from transform if necessary
            if features_data:
                log_features = features_data[0] # Log stats for the first trajectory part
            else:
                log_features = np.array([]) # Empty array if list is empty
        else:
            log_features = features_data

        if log_features.size > 0: # Check if there is data to compute stats on
            logger.info(f"Feature data summary (first 5 features if >5):")
            num_features_to_log = min(5, log_features.shape[1])
            logger.info(f"  Shape: {log_features.shape}")
            try:
                mean_vals = np.mean(log_features[:, :num_features_to_log], axis=0)
                std_vals = np.std(log_features[:, :num_features_to_log], axis=0)
                min_vals = np.min(log_features[:, :num_features_to_log], axis=0)
                max_vals = np.max(log_features[:, :num_features_to_log], axis=0)
                logger.info(f"  Mean (first {num_features_to_log}): {mean_vals}")
                logger.info(f"  Std Dev (first {num_features_to_log}): {std_vals}")
                logger.info(f"  Min (first {num_features_to_log}): {min_vals}")
                logger.info(f"  Max (first {num_features_to_log}): {max_vals}")
            except Exception as stat_err:
                logger.warning(f"Could not compute summary statistics for features: {stat_err}")
        else:
            logger.warning("Feature data is empty, cannot compute summary statistics.")
        # --- End logging summary statistics ---

        try:
            logger.info(f"Saving features data to '{output_features_data_file}'...")
            np.save(output_features_data_file, features_data)
            logger.info("Features data saved successfully.")
            # Add the check instruction here as well, for clarity
            logger.info(f"To inspect the full features data, load with: `import numpy as np; features = np.load('{output_features_data_file}')`")
        except Exception as e:
            logger.warning(f"Warning: Failed to save features data to '{output_features_data_file}'. Continuing analysis. Details: {e}")

        logger.info(f"Performing TICA calculation with lag={lag_time}...")
        if isinstance(features_data, list):
             if not features_data:
                 msg = "Feature data list is empty."
                 logger.error(msg)
                 raise ValueError(msg)
             features_data = features_data[0]

        actual_n_frames = features_data.shape[0]
        feature_dim = features_data.shape[1]

        max_tica_dim = min(feature_dim, actual_n_frames - lag_time -1 if actual_n_frames > lag_time + 1 else 0)

        if max_tica_dim < 1:
             msg = f"Cannot compute TICA. Insufficient effective samples ({actual_n_frames - lag_time - 1}) or feature dimensions ({feature_dim}). Max possible dim: {max_tica_dim}"
             logger.error(msg)
             raise ValueError(msg)

        if n_dim > max_tica_dim:
            logger.warning(f"Warning: Requested TICA dimensions ({n_dim}) exceeds maximum possible ({max_tica_dim}). Reducing to {max_tica_dim}.")
            n_dim = max_tica_dim

        tica_obj = coor.tica(features_data, lag=lag_time, dim=n_dim, kinetic_map=True)
        tica_output = tica_obj.get_output()[0]
        logger.info(f"TICA calculation complete. Output projection shape: {tica_output.shape}")
        logger.info(f"TICA Timescales: {tica_obj.timescales}")
        logger.info(f"TICA Eigenvalues: {tica_obj.eigenvalues}")

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
            logger.error(f"Error: Failed to save TICA object to '{output_tica_file}'. Cannot continue. Details: {e}")
            sys.exit(1)

        logger.info(f"Generating TICA plot...")
        actual_dims_computed = tica_output.shape[1]
        if actual_dims_computed == 0:
             logger.warning("Warning: TICA computation resulted in 0 dimensions. Cannot generate plot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_dim = min(2, actual_dims_computed)

            if plot_dim == 2:
                x = tica_output[:, 0]
                y = tica_output[:, 1]
                xlabel = 'TICA Component 1 (IC1)'
                ylabel = 'TICA Component 2 (IC2)'
                plot_title = f'TICA Projection (Lag: {lag_time} frames, Features: {feature_type})'

                if plot_type == 'scatter':
                    scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Frame Index')
                elif plot_type == 'hist2d':
                    try:
                        from matplotlib.colors import LogNorm
                        counts, xedges, yedges, image = ax.hist2d(
                            x, y, bins=100, cmap='viridis', cmin=1
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
                logger.info("Plotting TICA Component 1 vs Frame Index.")
                x = np.arange(len(tica_output))
                y = tica_output[:, 0]
                xlabel = 'Frame Index'
                ylabel = 'TICA Component 1'
                plot_title = f'TICA Component 1 (Lag: {lag_time} frames, Features: {feature_type})'
                ax.plot(x, y, marker='.', linestyle='-', markersize=2)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(output_plot_file, dpi=300)
            logger.info(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig)

    except FileNotFoundError:
        pass
    except MemoryError:
        logger.error(f"Error: Insufficient memory during TICA analysis. Consider using iterload or different features.")
        sys.exit(1)
    except ValueError as e:
         logger.error(f"Error during TICA analysis setup or calculation: {e}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during TICA analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logging.shutdown()

    print("--- TICA Analysis Finished ---")
    return tica_obj

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
        '--features', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'],
        default='backbone_torsions',
        help='Type of features to use for TICA. "backbone_torsions" is often faster.'
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