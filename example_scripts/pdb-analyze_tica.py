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
    import pyemma.coordinates as coor
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy, argparse.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)

def run_tica_analysis(pdb_file, lag_time, n_dim, output_file, feature_type):
    """
    Performs TICA analysis on a multi-frame PDB file using PyEMMA.

    Parameters
    ----------
    pdb_file : str
        Path to the input multi-frame PDB file.
    lag_time : int
        TICA lag time in units of frames.
    n_dim : int
        Number of TICA dimensions to compute.
    output_file : str
        Path to save the output TICA projection plot.
    feature_type : str
        Type of features to use ('ca_dist' or 'heavy_atom').
    """
    print("--- Starting TICA Analysis ---")
    print(f"Input PDB: {pdb_file}")
    print(f"Lag time: {lag_time} frames")
    print(f"Target TICA dimensions: {n_dim}")
    print(f"Feature type: {feature_type}")
    print(f"Output plot: {output_file}")

    if not os.path.exists(pdb_file):
        print(f"Error: Input PDB file not found at '{pdb_file}'", file=sys.stderr)
        sys.exit(1)

    try:
        # 1. Load topology using the first frame
        # MDTraj typically infers topology well from PDB.
        # We load the first frame specifically to get the topology object
        # required by the PyEMMA featurizer.
        print("Loading topology from the first frame...")
        try:
            topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology
        except Exception as e:
             print(f"Error: Failed to load topology from {pdb_file}. Check if it's a valid PDB format. Details: {e}", file=sys.stderr)
             sys.exit(1)

        if topology_mdtraj is None or topology_mdtraj.n_atoms == 0:
             raise ValueError("Could not load a valid topology from the PDB file.")
        print(f"Topology loaded: {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")


        # 2. Setup Featurizer
        print("Setting up featurizer...")
        feat = coor.featurizer(topology_mdtraj)
        features_data = None # Initialize feature data variable

        if feature_type == 'ca_dist':
            # Select C-alpha atoms
            ca_indices = topology_mdtraj.select('name CA')
            if len(ca_indices) < 2:
                raise ValueError(f"Not enough C-alpha atoms (< 2) found in the topology ({len(ca_indices)} found). Cannot compute C-alpha distances.")
            
            print("Loading full trajectory for C-alpha distance processing...")
            traj = md.load(pdb_file, top=topology_mdtraj)
            print(f"Loaded trajectory with {traj.n_frames} frames.")
            if traj.n_frames <= lag_time:
                print(f"Warning: Trajectory length ({traj.n_frames}) is less than or equal to the lag time ({lag_time}). TICA results may be unreliable.", file=sys.stderr)
            
            feat.add_distances_ca()
            print(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")
            print(f"Computing C-alpha distance features...")
            features_data = feat.transform(traj)

        elif feature_type == 'heavy_atom':
            # Select non-hydrogen atoms
            heavy_atom_indices = topology_mdtraj.select('not element H')
            if len(heavy_atom_indices) == 0:
                 raise ValueError("No heavy atoms found in the topology.")

            # Heavy atom coordinates require alignment for meaningful TICA
            print("Loading full trajectory for heavy atom processing...")
            # Load the entire trajectory into memory for alignment
            traj = md.load(pdb_file, top=topology_mdtraj)
            print(f"Loaded trajectory with {traj.n_frames} frames.")
            if traj.n_frames <= lag_time:
                print(f"Warning: Trajectory length ({traj.n_frames}) is less than or equal to the lag time ({lag_time}). TICA results may be unreliable.", file=sys.stderr)
            if traj.n_frames < 2:
                 raise ValueError("Trajectory has fewer than 2 frames, cannot perform alignment or TICA.")

            print("Aligning trajectory to the first frame (using heavy atoms)...")
            # Align based on heavy atoms to remove global rotation/translation
            traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)

            print("Adding heavy atom coordinates to featurizer...")
            feat.add_selection(heavy_atom_indices)
            print(f"Using aligned heavy atom coordinates. Feature dimension: {feat.dimension()}")

            # Calculate features from the *aligned* in-memory trajectory object
            print("Calculating features from aligned trajectory (this might take time and memory)...")
            features_data = feat.transform(traj)
            # Note: This requires loading the whole trajectory into memory.
            # For very large files, consider pre-aligning the PDB externally.

        if features_data is None:
             raise RuntimeError("Feature data was not generated correctly.")

        # 3. Perform TICA
        print(f"Performing TICA calculation with lag={lag_time}...")
        # Ensure requested dimensions don't exceed feature dimensions or n_frames - lag - 1
        max_tica_dim = min(feat.dimension(), len(features_data[0]) - lag_time -1 if isinstance(features_data, list) else features_data.shape[0] - lag_time - 1)

        if max_tica_dim < 1:
             raise ValueError(f"Cannot compute TICA. Insufficient effective samples or feature dimensions relative to lag time. (Max possible dim: {max_tica_dim}, Feature dim: {feat.dimension()}, Frames: {len(features_data[0]) if isinstance(features_data, list) else features_data.shape[0]}, Lag: {lag_time})")

        if n_dim > max_tica_dim:
            print(f"Warning: Requested TICA dimensions ({n_dim}) exceeds the maximum possible ({max_tica_dim}) based on data length, lag time and feature dimension. Reducing dimensions to {max_tica_dim}.", file=sys.stderr)
            n_dim = max_tica_dim


        tica = coor.tica(features_data, lag=lag_time, dim=n_dim)
        # get_output() returns a list (one element per trajectory source)
        tica_output = tica.get_output()[0]
        print(f"TICA calculation complete. Output shape: {tica_output.shape}") # Shape is (n_frames, n_tica_dimensions)

        # Check if enough dimensions were actually computed
        actual_dims = tica_output.shape[1]
        if actual_dims == 0:
             raise ValueError("TICA computation resulted in 0 dimensions. Cannot proceed.")
        if actual_dims < 2:
            print("Warning: Fewer than 2 TICA dimensions were computed. Plot will show TICA 1 vs Frame Index.", file=sys.stderr)

        # 4. Plotting
        print(f"Generating TICA plot...")
        fig, ax = plt.subplots(figsize=(8, 6))

        if actual_dims >= 2:
            x = tica_output[:, 0]
            y = tica_output[:, 1]
            xlabel = 'TICA Component 1'
            ylabel = 'TICA Component 2'
        else: # Only 1 dimension computed
            x = np.arange(len(tica_output)) # Frame index
            y = tica_output[:, 0]           # TICA 1 value
            xlabel = 'Frame Index'
            ylabel = 'TICA Component 1'

        # Create a 2D histogram (density plot) - often better for visualizing distributions
        try:
            # Use LogNorm for better color contrast if counts vary widely
            from matplotlib.colors import LogNorm
            counts, xedges, yedges, image = ax.hist2d(
                x, y, bins=100, cmap='viridis', cmin=1 # cmin=1 avoids plotting empty bins
                # norm=LogNorm() # Uncomment for logarithmic color scale
            )
            cbar = fig.colorbar(image, ax=ax)
            cbar.set_label('Counts')
        except ValueError as e:
            # Fallback to scatter plot if hist2d fails (e.g., data range issues)
            print(f"Warning: Could not generate density plot (hist2d). Error: {e}. Falling back to scatter plot.", file=sys.stderr)
            # Color points by time/frame index
            scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Frame Index')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'TICA Projection (Lag: {lag_time} frames, Features: {feature_type})')
        plt.savefig(output_file, dpi=300)
        print(f"TICA plot saved to '{output_file}'")
        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError:
        # Error already printed before try block
        pass
    except MemoryError:
        print(f"Error: Insufficient memory. This can happen when loading large trajectories or using high-dimensional features (like heavy_atom). Try reducing data size or using features like 'ca_dist'.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"Error: Data processing error. {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print detailed traceback for unexpected errors
        sys.exit(1)

    print("--- TICA Analysis Finished ---")


def main():
    """
    Parses command line arguments and runs the TICA analysis.
    """
    parser = argparse.ArgumentParser(
        description='Perform Time-lagged Independent Component Analysis (TICA) on a multi-frame PDB file using PyEMMA.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    parser.add_argument(
        'pdb_file',
        help='Path to the input multi-frame PDB file.'
    )
    parser.add_argument(
        '-l', '--lag',
        type=int,
        default=10,
        help='TICA lag time (in frames). Represents the time scale of interest.'
    )
    parser.add_argument(
        '-d', '--dim',
        type=int,
        default=2,
        help='Number of TICA dimensions to calculate. Must be >= 1.'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output filename for the 2D TICA projection plot. If not specified, will save to input file directory as "tica_projection.png".'
    )
    parser.add_argument(
        '--features',
        choices=['ca_dist', 'heavy_atom'],
        default='ca_dist',
        help='Type of features to use for TICA. '
             '"ca_dist": C-alpha distances (robust to rotation/translation). '
             '"heavy_atom": Coordinates of non-hydrogen atoms (requires alignment; the script will align in memory, which can be slow/memory-intensive for large files).'
    )

    # Handle cases where no arguments are provided or --help is used
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Basic input validation
    if args.lag <= 0:
        print("Error: Lag time (--lag) must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if args.dim < 1:
        print("Error: Number of TICA dimensions (--dim) must be at least 1.", file=sys.stderr)
        sys.exit(1)
    
    # Set default output path to input file's directory if not specified
    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(args.pdb_file))
        args.output = os.path.join(input_dir, 'tica_projection.png')
        print(f"No output path specified. Results will be saved to: {args.output}")

    # Run the main analysis function
    run_tica_analysis(args.pdb_file, args.lag, args.dim, args.output, args.features)


if __name__ == "__main__":
    main()
