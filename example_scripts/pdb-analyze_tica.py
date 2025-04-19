#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings

# Suppress specific warnings from MDTraj and PyEMMA if desired
warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyemma')

try:
    import mdtraj as md
    import pyemma                          # Import top-level pyemma for saving/loading
    import pyemma.coordinates as coor
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy, argparse.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)

def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type):
    """
    Performs TICA analysis on a multi-frame PDB file using PyEMMA,
    saves the TICA object and projection plot to the specified directory.

    Parameters
    ----------
    pdb_file : str
        Path to the input multi-frame PDB file.
    lag_time : int
        TICA lag time in units of frames.
    n_dim : int
        Number of TICA dimensions to compute.
    output_dir : str
        Directory where the plot and TICA object will be saved.
    output_basename : str
        Base name for the output files (plot and TICA object).
    feature_type : str
        Type of features to use ('ca_dist' or 'heavy_atom').

    Returns
    -------
    tica_obj : pyemma.coordinates.TICA or None
        The computed TICA object, or None if an error occurred before computation.
    """
    print("--- Starting TICA Analysis ---")
    print(f"Input PDB: {pdb_file}")
    print(f"Lag time: {lag_time} frames")
    print(f"Target TICA dimensions: {n_dim}")
    print(f"Feature type: {feature_type}")
    print(f"Output directory: {output_dir}")

    # Construct full output paths
    plot_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}.png"
    tica_obj_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_model.pyemma"
    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_file = os.path.join(output_dir, tica_obj_filename)

    print(f"Output plot file: {output_plot_file}")
    print(f"Output TICA object file: {output_tica_file}")


    tica_obj = None # Initialize return value

    if not os.path.exists(pdb_file):
        print(f"Error: Input PDB file not found at '{pdb_file}'", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load topology
        print("Loading topology from the first frame...")
        try:
            topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology
        except Exception as e:
             print(f"Error: Failed to load topology from {pdb_file}. Check format. Details: {e}", file=sys.stderr)
             sys.exit(1)

        if topology_mdtraj is None or topology_mdtraj.n_atoms == 0:
             raise ValueError("Could not load a valid topology from the PDB file.")
        print(f"Topology loaded: {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")


        # 2. Setup Featurizer
        print("Setting up featurizer...")
        feat = coor.featurizer(topology_mdtraj)
        features_data = None

        # --- Feature selection and calculation (same as before) ---
        if feature_type == 'ca_dist':
            ca_indices = topology_mdtraj.select('name CA and protein') # Be more specific
            if len(ca_indices) < 2:
                raise ValueError(f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found). Cannot compute C-alpha distances.")
            print("Loading full trajectory for C-alpha distance processing...")
            traj = md.load(pdb_file, top=topology_mdtraj)
            print(f"Loaded trajectory with {traj.n_frames} frames.")
            if traj.n_frames <= lag_time:
                print(f"Warning: Trajectory length ({traj.n_frames}) <= lag time ({lag_time}). TICA unreliable.", file=sys.stderr)
            feat.add_distances_ca()
            print(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")
            print(f"Computing C-alpha distance features...")
            features_data = feat.transform(traj)

        elif feature_type == 'heavy_atom':
            heavy_atom_indices = topology_mdtraj.select('not element H and protein') # Be more specific
            if len(heavy_atom_indices) == 0:
                 raise ValueError("No heavy atoms found in protein residues.")
            print("Loading full trajectory for heavy atom processing...")
            traj = md.load(pdb_file, top=topology_mdtraj)
            print(f"Loaded trajectory with {traj.n_frames} frames.")
            if traj.n_frames <= lag_time:
                print(f"Warning: Trajectory length ({traj.n_frames}) <= lag time ({lag_time}). TICA unreliable.", file=sys.stderr)
            if traj.n_frames < 2:
                 raise ValueError("Trajectory has < 2 frames, cannot align/TICA.")
            print("Aligning trajectory to the first frame (using heavy atoms)...")
            traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)
            print("Adding heavy atom coordinates to featurizer...")
            feat.add_selection(heavy_atom_indices)
            print(f"Using aligned heavy atom coordinates. Feature dimension: {feat.dimension()}")
            print("Calculating features from aligned trajectory...")
            features_data = feat.transform(traj)
        # --- End feature selection ---

        if features_data is None:
             raise RuntimeError("Feature data was not generated correctly.")

        # 3. Perform TICA
        print(f"Performing TICA calculation with lag={lag_time}...")
        actual_n_frames = features_data.shape[0] if isinstance(features_data, np.ndarray) \
                          else (len(features_data[0]) if (isinstance(features_data, list) and features_data) else 0)

        max_tica_dim = min(feat.dimension(), actual_n_frames - lag_time -1 if actual_n_frames > lag_time + 1 else 0)

        if max_tica_dim < 1:
             raise ValueError(f"Cannot compute TICA. Insufficient effective samples or feature dimensions. (Max possible dim: {max_tica_dim}, Feature dim: {feat.dimension()}, Frames: {actual_n_frames}, Lag: {lag_time})")

        if n_dim > max_tica_dim:
            print(f"Warning: Requested TICA dimensions ({n_dim}) exceeds maximum possible ({max_tica_dim}). Reducing to {max_tica_dim}.", file=sys.stderr)
            n_dim = max_tica_dim


        tica_obj = coor.tica(features_data, lag=lag_time, dim=n_dim) # Assign to tica_obj
        tica_output = tica_obj.get_output()[0]
        print(f"TICA calculation complete. Output shape: {tica_output.shape}")

        #------------------------------------#
        # 4. SAVE THE TICA OBJECT            #
        #------------------------------------#
        try:
            print(f"Saving TICA object to '{output_tica_file}'...")
            # Use PyEMMA's built-in save method
            tica_obj.save(output_tica_file, overwrite=True) # Overwrite if file exists
            print("TICA object saved successfully.")
        except Exception as e:
            print(f"Error: Failed to save TICA object to '{output_tica_file}'. Details: {e}", file=sys.stderr)
            # Decide if this should be fatal or just a warning
            # sys.exit(1) # Make it fatal

        # 5. Plotting
        print(f"Generating TICA plot...")
        actual_dims = tica_output.shape[1]
        if actual_dims == 0:
             print("Error: TICA computation resulted in 0 dimensions. Cannot generate plot.", file=sys.stderr)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            if actual_dims >= 2:
                x = tica_output[:, 0]
                y = tica_output[:, 1]
                xlabel = 'TICA Component 1 (IC1)'
                ylabel = 'TICA Component 2 (IC2)'
            else: # Only 1 dimension computed
                print("Warning: Only 1 TICA dimension computed. Plotting TICA 1 vs Frame Index.", file=sys.stderr)
                x = np.arange(len(tica_output)) # Frame index
                y = tica_output[:, 0]           # TICA 1 value
                xlabel = 'Frame Index'
                ylabel = 'TICA Component 1'

            try:
                from matplotlib.colors import LogNorm
                counts, xedges, yedges, image = ax.hist2d(
                    x, y, bins=100, cmap='viridis', cmin=1
                    # norm=LogNorm()
                )
                cbar = fig.colorbar(image, ax=ax)
                cbar.set_label('Counts')
            except ValueError as e:
                print(f"Warning: Could not generate hist2d plot. Error: {e}. Falling back to scatter plot.", file=sys.stderr)
                scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Frame Index')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'TICA Projection (Lag: {lag_time} frames, Features: {feature_type})')
            plt.savefig(output_plot_file, dpi=300)
            print(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig)

    except FileNotFoundError:
        pass
    except MemoryError:
        print(f"Error: Insufficient memory. Try 'ca_dist' features or pre-process data.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"Error: Data processing or value error. {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during TICA analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("--- TICA Analysis Finished ---")
    return tica_obj # Return the object (or None if error)


def main():
    """
    Parses command line arguments and runs the TICA analysis.
    """
    parser = argparse.ArgumentParser(
        description='Perform TICA on a PDB trajectory, save the TICA object and projection plot to a specified directory.',
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
        '-o', '--output-dir', default="tica_results", # Changed default
        help='Directory to save the TICA projection plot and TICA object file.'
    )
    parser.add_argument(
        '--features', choices=['ca_dist', 'heavy_atom'], default='ca_dist',
        help='Type of features to use for TICA.'
    )
    parser.add_argument(
        '--basename', default=None,
        help='Basename for output files. If not specified, it uses the input PDB filename without extension.'
    )


    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Input Validation ---
    if args.lag <= 0:
        print("Error: Lag time (--lag) must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.dim < 1:
        print("Error: TICA dimensions (--dim) must be >= 1.", file=sys.stderr)
        sys.exit(1)

    # --- Determine Output Basename ---
    if args.basename is None:
        args.basename = os.path.splitext(os.path.basename(args.pdb_file))[0]
        print(f"Using input filename base for output files: '{args.basename}'")


    # --- Run Analysis ---
    run_tica_analysis(
        pdb_file=args.pdb_file,
        lag_time=args.lag,
        n_dim=args.dim,
        output_dir=args.output_dir, # Pass the directory
        output_basename=args.basename, # Pass the base name for files
        feature_type=args.features
    )

if __name__ == "__main__":
    main()
