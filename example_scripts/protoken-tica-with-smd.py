#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress specific warnings if desired
warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyemma')

try:
    import mdtraj as md
    import pyemma
    import pyemma.coordinates as coor
except ImportError as e:
    print(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)

def load_trajectory_and_featurize(pdb_file, feature_type):
    """
    Loads a trajectory PDB and calculates specified features using PyEMMA.
    Handles alignment for 'heavy_atom' features.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file (can be multi-frame).
    feature_type : str
        'ca_dist' or 'heavy_atom'. Must match the method used for the TICA model.

    Returns
    -------
    features_data : np.ndarray or None
        The calculated features, or None if an error occurred.
    """
    print(f"\n--- Featurizing '{os.path.basename(pdb_file)}' (Type: {feature_type}) ---")
    if not os.path.exists(pdb_file):
        print(f"Error: Input PDB file not found at '{pdb_file}'", file=sys.stderr)
        return None # Return None on error

    features_data = None
    try:
        # 1. Load topology first (needed for featurizer setup)
        print("Loading topology from first frame...")
        try:
            # frame=0 ensures only topology is read initially
            topology_mdtraj = md.load_pdb(pdb_file, frame=0).topology
        except Exception as e:
             print(f"Error: Failed to load topology from {pdb_file}. Details: {e}", file=sys.stderr)
             return None

        if topology_mdtraj is None or topology_mdtraj.n_atoms == 0:
             raise ValueError("Could not load a valid topology.")
        print(f"Topology loaded: {topology_mdtraj.n_atoms} atoms, {topology_mdtraj.n_residues} residues.")

        # 2. Setup Featurizer
        print("Setting up featurizer...")
        feat = coor.featurizer(topology_mdtraj)

        # 3. Load full trajectory and apply featurization + necessary pre-processing
        print("Loading full trajectory...")
        # Load the whole trajectory as MDTraj object
        traj = md.load(pdb_file, top=topology_mdtraj)
        print(f"Loaded trajectory with {traj.n_frames} frames.")
        if traj.n_frames == 0:
            raise ValueError("Trajectory contains no frames.")

        if feature_type == 'ca_dist':
            ca_indices = topology_mdtraj.select('name CA and protein')
            if len(ca_indices) < 2:
                raise ValueError(f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found).")
            feat.add_distances_ca()
            print(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")
            print(f"Calculating C-alpha distance features...")
            features_data = feat.transform(traj)

        elif feature_type == 'heavy_atom':
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if len(heavy_atom_indices) == 0:
                 raise ValueError("No heavy atoms found in protein residues.")
            if traj.n_frames < 2:
                 print("Warning: Trajectory has < 2 frames, cannot perform alignment. Using unaligned coordinates.", file=sys.stderr)
            else:
                print("Aligning trajectory to the first frame (using heavy atoms)...")
                traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)
            feat.add_selection(heavy_atom_indices)
            print(f"Using aligned heavy atom coordinates. Feature dimension: {feat.dimension()}")
            print("Calculating features from aligned trajectory...")
            features_data = feat.transform(traj)
        else:
            # Should not happen due to argparse choices, but good practice
            raise ValueError(f"Invalid feature_type specified: {feature_type}")

        if features_data is None:
             raise RuntimeError("Feature data was not generated correctly.")

        print(f"Featurization successful. Feature shape: {features_data.shape}")

    except FileNotFoundError:
         # Already handled at the start
         pass
    except ValueError as e:
         print(f"Error during featurization of '{pdb_file}': {e}", file=sys.stderr)
         return None
    except MemoryError:
        print(f"Error: Insufficient memory during featurization of '{pdb_file}'.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during featurization of '{pdb_file}': {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

    print(f"--- Finished Featurizing '{os.path.basename(pdb_file)}' ---")
    return features_data


def main():
    parser = argparse.ArgumentParser(
        description='Project SMD and ProToken trajectories onto a pre-computed TICA model and plot comparison.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'tica_model_file',
        help='Path to the saved TICA object file (*.pyemma).'
    )
    parser.add_argument(
        'smd_pdb_file',
        help='Path to the original SMD trajectory PDB file (used to train the TICA model).'
    )
    parser.add_argument(
        'protoken_pdb_file',
        help='Path to the ProToken trajectory PDB file to project and compare.'
    )
    parser.add_argument(
        '--feature_type',
        choices=['ca_dist', 'heavy_atom'],
        required=True, # User MUST specify how the TICA model was built
        help='Type of features used to build the TICA model AND to featurize the trajectories for projection.'
    )
    parser.add_argument(
        '-o', '--output_plot',
        default=None,
        help='Output filename for the comparison plot. Defaults to TICA model dir + "tica_comparison.png".'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

     # --- Basic Input Validation ---
    if not os.path.exists(args.tica_model_file):
        print(f"Error: TICA model file not found: '{args.tica_model_file}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.smd_pdb_file):
        print(f"Error: SMD PDB file not found: '{args.smd_pdb_file}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.protoken_pdb_file):
        print(f"Error: ProToken PDB file not found: '{args.protoken_pdb_file}'", file=sys.stderr)
        sys.exit(1)

    # --- Set Default Output Path ---
    if args.output_plot is None:
        output_dir = os.path.dirname(os.path.abspath(args.tica_model_file))
        args.output_plot = os.path.join(output_dir, 'tica_comparison_plot.png')
        print(f"No output plot path specified. Plot will be saved to: {args.output_plot}")
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    try:
        # 1. Load the saved TICA model
        print(f"Loading TICA model from: {args.tica_model_file}")
        tica_transformer = pyemma.load(args.tica_model_file)
        if not isinstance(tica_transformer, coor.TICA):
             raise TypeError(f"Loaded object is not a PyEMMA TICA object ({type(tica_transformer)}).")
        print(f"TICA model loaded successfully. Expected feature dimension: {tica_transformer.dimension}, Output dimensions: {tica_transformer.output_dimension()}")
        n_tica_dims = tica_transformer.output_dimension()
        if n_tica_dims < 1:
             raise ValueError("Loaded TICA model has 0 output dimensions.")


        # 2. Featurize and Project SMD Trajectory
        smd_features = load_trajectory_and_featurize(args.smd_pdb_file, args.feature_type)
        if smd_features is None:
            raise RuntimeError(f"Failed to featurize SMD trajectory: {args.smd_pdb_file}")

        # Verify feature dimension consistency
        if smd_features.shape[1] != tica_transformer.feature_dimension:
              print(f"Warning: Feature dimension mismatch! TICA model expects {tica_transformer.feature_dimension} features, "
                    f"but SMD PDB featurization resulted in {smd_features.shape[1]} features. "
                    f"Ensure --feature_type matches model creation and PDBs are compatible.", file=sys.stderr)
              # Continue, but transform might fail or give meaningless results.

        print("Projecting SMD features onto loaded TICA model...")
        Y_smd = tica_transformer.transform(smd_features)
        print(f"SMD projection successful. Shape: {Y_smd.shape}")

        # 3. Featurize and Project ProToken Trajectory
        pt_features = load_trajectory_and_featurize(args.protoken_pdb_file, args.feature_type)
        if pt_features is None:
            raise RuntimeError(f"Failed to featurize ProToken trajectory: {args.protoken_pdb_file}")

        # Verify feature dimension consistency
        if pt_features.shape[1] != tica_transformer.feature_dimension:
              print(f"Warning: Feature dimension mismatch! TICA model expects {tica_transformer.feature_dimension} features, "
                    f"but ProToken PDB featurization resulted in {pt_features.shape[1]} features. "
                    f"Ensure --feature_type matches model creation and PDBs are compatible.", file=sys.stderr)

        print("Projecting ProToken features onto loaded TICA model...")
        Y_pt = tica_transformer.transform(pt_features)
        print(f"ProToken projection successful. Shape: {Y_pt.shape}")


        # 4. Plotting Comparison
        print(f"Generating comparison plot...")
        fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger figure

        plot_dim = min(2, n_tica_dims) # We can only plot 1D or 2D

        if plot_dim == 2:
            xlabel = 'TICA Component 1 (IC1)'
            ylabel = 'TICA Component 2 (IC2)'
            # Plot SMD trajectory (background density or scatter)
            try:
                from matplotlib.colors import LogNorm
                counts, xedges, yedges, img = ax.hist2d(Y_smd[:, 0], Y_smd[:, 1], bins=100, cmap="Blues", cmin=1) # Background density
                # cbar = fig.colorbar(img, ax=ax, label="SMD Counts") # Optional colorbar for density
            except Exception: # Fallback if hist2d fails
                 ax.scatter(Y_smd[:, 0], Y_smd[:, 1], alpha=0.15, s=10, color="lightblue", label='SMD (Sampled Points)', zorder=1) # Scatter

            # Plot ProToken trajectory (foreground line+markers)
            ax.plot(Y_pt[:, 0], Y_pt[:, 1], marker='o', markersize=4, linestyle='-', linewidth=1.5, color='red', label='ProToken Path', zorder=2)
            # Optionally add start/end points for ProToken
            ax.scatter(Y_pt[0, 0], Y_pt[0, 1], marker='^', s=100, color='green', label='ProToken Start', zorder=3, edgecolors='black')
            ax.scatter(Y_pt[-1, 0], Y_pt[-1, 1], marker='s', s=100, color='purple', label='ProToken End', zorder=3, edgecolors='black')

        elif plot_dim == 1:
             xlabel = 'Frame Index'
             ylabel = 'TICA Component 1 (IC1)'
             ax.plot(np.arange(len(Y_smd)), Y_smd[:, 0], alpha=0.5, color="lightblue", label='SMD IC1')
             ax.plot(np.arange(len(Y_pt)), Y_pt[:, 0], marker='o', markersize=4, linestyle='-', color='red', label='ProToken IC1')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Trajectory Comparison in TICA Space (Features: {args.feature_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout() # Adjust layout
        plt.savefig(args.output_plot, dpi=300)
        print(f"Comparison plot saved to '{args.output_plot}'")
        plt.close(fig)


    except FileNotFoundError:
        # Error printed earlier or by pyemma.load
        sys.exit(1)
    except (TypeError, ValueError, RuntimeError, IndexError) as e:
        # Catch common errors during processing
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
