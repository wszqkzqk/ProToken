#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyemma')

try:
    import mdtraj as md
    import pyemma                          # Import top-level pyemma for saving/loading
    import pyemma.coordinates as coor
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy, argparse.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)

def run_tica_analysis(pdb_file, lag_time, n_dim, output_dir, output_basename, feature_type, plot_type):
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
        Type of features to use ('ca_dist', 'heavy_atom', 'backbone_torsions').
    plot_type : str
        Type of plot ('scatter' or 'hist2d').

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
    print(f"Plot type: {plot_type}")
    print(f"Output directory: {output_dir}")

    # Construct full output paths
    plot_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}.png"
    tica_obj_filename = f"{output_basename}_tica_lag{lag_time}_dim{n_dim}_{feature_type}_model.pyemma"
    output_plot_file = os.path.join(output_dir, plot_filename)
    output_tica_file = os.path.join(output_dir, tica_obj_filename)

    print(f"Output plot file: {output_plot_file}")
    print(f"Output TICA object file: {output_tica_file}")

    tica_obj = None # Initialize return value

    if not os.path.exists(pdb_file):
        print(f"Error: Input PDB file not found at '{pdb_file}'", file=sys.stderr)
        sys.exit(1) # Exit if input file not found

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load topology
        print("Loading topology from the first frame...")
        try:
            # Load topology using only the first frame to save memory
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
        needs_alignment = False # Flag to check if alignment is needed

        # --- Feature selection ---
        if feature_type == 'ca_dist':
            # Select protein C-alpha atoms
            ca_indices = topology_mdtraj.select('name CA and protein')
            if len(ca_indices) < 2:
                raise ValueError(f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found).")
            feat.add_distances_ca()
            print(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")

        elif feature_type == 'heavy_atom':
            # Select protein non-hydrogen atoms
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if len(heavy_atom_indices) == 0:
                 raise ValueError("No heavy atoms found in protein residues.")
            feat.add_selection(heavy_atom_indices)
            needs_alignment = True # Heavy atom coords need alignment
            print(f"Using heavy atom coordinates. Feature dimension: {feat.dimension()}")

        elif feature_type == 'backbone_torsions':
            # PyEMMA selects backbone atoms automatically
            feat.add_backbone_torsions(cossin=True, periodic=False) # Use cossin for TICA
            print(f"Using backbone torsions (cos/sin). Feature dimension: {feat.dimension()}")
            # Check if any torsions were actually added
            if feat.dimension() == 0:
                raise ValueError("Could not find any backbone torsions. Is it a valid protein structure?")

        else:
             # Should not happen due to argparse choices
             raise ValueError(f"Invalid feature type '{feature_type}' specified.")
        # --- End feature selection ---

        # 3. Load Trajectory & Calculate Features
        print("Loading full trajectory...")
        # Use iterload for potentially large files if memory is an issue,
        # but TICA often benefits from having all data. We'll load fully here.
        traj = md.load(pdb_file, top=topology_mdtraj)
        print(f"Loaded trajectory with {traj.n_frames} frames.")

        if traj.n_frames == 0:
             raise ValueError("Input PDB file contains 0 frames.")
        if traj.n_frames <= lag_time:
            print(f"Warning: Trajectory length ({traj.n_frames}) <= lag time ({lag_time}). TICA results might be unreliable or fail.", file=sys.stderr)
            if traj.n_frames <= 1: # Cannot compute TICA if frames <= 1
                 raise ValueError("Cannot compute TICA with <= 1 frame.")

        # Perform alignment if needed (only for heavy_atom features)
        if needs_alignment:
            if traj.n_frames < 2:
                 print("Warning: Only 1 frame in trajectory, cannot perform alignment.", file=sys.stderr)
            else:
                 print("Aligning trajectory to the first frame (using heavy atoms)...")
                 # Use the previously selected heavy atom indices for alignment
                 heavy_atom_indices = topology_mdtraj.select('not element H and protein')
                 traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)

        # Calculate features from the (potentially aligned) trajectory
        print(f"Calculating features ({feature_type})...")
        features_data = feat.transform(traj) # Use the in-memory trajectory

        if features_data is None or features_data.shape[0] == 0:
             raise RuntimeError("Feature calculation resulted in empty data.")
        print(f"Feature calculation complete. Feature shape: {features_data.shape}")

        # 4. Perform TICA
        print(f"Performing TICA calculation with lag={lag_time}...")
        # Ensure data is numpy array (transform usually returns this)
        if isinstance(features_data, list):
             if not features_data: raise ValueError("Feature data list is empty.")
             features_data = features_data[0] # Assuming single trajectory input

        actual_n_frames = features_data.shape[0]
        feature_dim = features_data.shape[1]

        # Recalculate max possible dimension based on actual frames and feature dim
        max_tica_dim = min(feature_dim, actual_n_frames - lag_time -1 if actual_n_frames > lag_time + 1 else 0)

        if max_tica_dim < 1:
             raise ValueError(f"Cannot compute TICA. Insufficient effective samples ({actual_n_frames - lag_time - 1}) or feature dimensions ({feature_dim}). Max possible dim: {max_tica_dim}")

        if n_dim > max_tica_dim:
            print(f"Warning: Requested TICA dimensions ({n_dim}) exceeds maximum possible ({max_tica_dim}). Reducing to {max_tica_dim}.", file=sys.stderr)
            n_dim = max_tica_dim # Adjust n_dim

        tica_obj = coor.tica(features_data, lag=lag_time, dim=n_dim, kinetic_map=True) # Enable kinetic map
        tica_output = tica_obj.get_output()[0]
        print(f"TICA calculation complete. Output shape: {tica_output.shape}")
        print(f"TICA Timescales: {tica_obj.timescales}")
        print(f"TICA Eigenvalues: {tica_obj.eigenvalues}")


        # 5. SAVE THE TICA OBJECT
        try:
            print(f"Saving TICA object to '{output_tica_file}'...")
            tica_obj.save(output_tica_file, overwrite=True)
            print("TICA object saved successfully.")
        except Exception as e:
            # Make failure to save fatal, as the next script depends on it
            print(f"Error: Failed to save TICA object to '{output_tica_file}'. Cannot continue. Details: {e}", file=sys.stderr)
            sys.exit(1)

        # 6. Plotting
        print(f"Generating TICA plot...")
        actual_dims_computed = tica_output.shape[1]
        if actual_dims_computed == 0:
             print("Warning: TICA computation resulted in 0 dimensions. Cannot generate plot.", file=sys.stderr)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_dim = min(2, actual_dims_computed) # Decide whether to plot 1D or 2D

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
                        # Adjust bins based on data range or keep fixed? Fixed for now.
                        counts, xedges, yedges, image = ax.hist2d(
                            x, y, bins=100, cmap='viridis', cmin=1 # Show bins with at least one count
                            # norm=LogNorm() # Uncomment for log scale if needed
                        )
                        cbar = fig.colorbar(image, ax=ax)
                        cbar.set_label('Counts')
                    except ValueError as e:
                        print(f"Warning: hist2d plot failed: {e}. Check data range/variance.", file=sys.stderr)
                        # Fallback to scatter if hist2d fails
                        scatter = ax.scatter(x, y, s=5, alpha=0.5, c=np.arange(len(tica_output)), cmap='viridis')
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label('Frame Index')
                        plot_title += " (hist2d failed, showing scatter)"
                else:
                    raise ValueError(f"Invalid plot_type: {plot_type}") # Should be caught by argparse

            elif plot_dim == 1:
                print("Plotting TICA Component 1 vs Frame Index.")
                x = np.arange(len(tica_output)) # Frame index
                y = tica_output[:, 0]           # TICA 1 value
                xlabel = 'Frame Index'
                ylabel = 'TICA Component 1'
                plot_title = f'TICA Component 1 (Lag: {lag_time} frames, Features: {feature_type})'
                ax.plot(x, y, marker='.', linestyle='-', markersize=2) # Simple line plot

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout() # Adjust layout
            plt.savefig(output_plot_file, dpi=300)
            print(f"TICA plot saved to '{output_plot_file}'")
            plt.close(fig) # Close figure to free memory

    except FileNotFoundError: # Should be caught earlier
        pass
    except MemoryError:
        print(f"Error: Insufficient memory during TICA analysis. Consider using iterload or different features.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"Error during TICA analysis setup or calculation: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during TICA analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("--- TICA Analysis Finished ---")
    return tica_obj

def main():
    parser = argparse.ArgumentParser(
        description='Perform TICA on a PDB trajectory, save the TICA object and projection plot.',
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
        '-o', '--output-dir', default="tica_analysis_output", # Changed default name
        help='Directory to save the TICA projection plot and TICA object file.'
    )
    parser.add_argument(
        '--features', choices=['ca_dist', 'heavy_atom', 'backbone_torsions'], # Added backbone_torsions
        default='backbone_torsions', # Changed default to faster option
        help='Type of features to use for TICA. "backbone_torsions" is often faster.'
    )
    parser.add_argument(
        '--plot-type', choices=['scatter', 'hist2d'], default='scatter', # Changed default
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
        print(f"Using input filename base for output files: '{args.basename}'")

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