#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import warnings

# Set backend before importing pyplot to avoid GUI issues on servers
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyemma')

try:
    import mdtraj as md
    import pyemma # For loading the model
    import pyemma.coordinates as coor
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing required library. Please install PyEMMA, MDTraj, Matplotlib, NumPy.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)

# --- load_trajectory_and_featurize function remains the same as in the previous good version ---
# (Make sure you have the version that correctly handles alignment reference)
def load_trajectory_and_featurize(pdb_file, feature_type, reference_traj_for_alignment=None):
    """
    Loads a trajectory PDB and calculates specified features using PyEMMA.
    Handles alignment for 'heavy_atom' features, aligning to a reference if provided.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB file (can be multi-frame).
    feature_type : str
        'ca_dist', 'heavy_atom', or 'backbone_torsions'. Must match the TICA model.
    reference_traj_for_alignment : mdtraj.Trajectory, optional
        Reference trajectory (usually the first frame of SMD) for alignment if feature_type is 'heavy_atom'.

    Returns
    -------
    features_data : np.ndarray or None
        The calculated features, or None if an error occurred.
    """
    print(f"\n--- Featurizing '{os.path.basename(pdb_file)}' (Type: {feature_type}) ---")
    if not os.path.exists(pdb_file):
        print(f"Error: Input PDB file not found at '{pdb_file}'", file=sys.stderr)
        return None

    features_data = None
    try:
        # 1. Load topology first
        print("Loading topology from first frame...")
        try:
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
        needs_alignment = False

        # --- Feature selection ---
        if feature_type == 'ca_dist':
            ca_indices = topology_mdtraj.select('name CA and protein')
            if len(ca_indices) < 2:
                raise ValueError(f"Not enough protein C-alpha atoms (< 2) found ({len(ca_indices)} found).")
            feat.add_distances_ca()
            print(f"Using C-alpha distances. Feature dimension: {feat.dimension()}")

        elif feature_type == 'heavy_atom':
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if len(heavy_atom_indices) == 0:
                 raise ValueError("No heavy atoms found in protein residues.")
            feat.add_selection(heavy_atom_indices)
            needs_alignment = True
            print(f"Using heavy atom coordinates. Feature dimension: {feat.dimension()}")

        elif feature_type == 'backbone_torsions':
            feat.add_backbone_torsions(cossin=True, periodic=False)
            print(f"Using backbone torsions (cos/sin). Feature dimension: {feat.dimension()}")
            if feat.dimension() == 0:
                raise ValueError("Could not find any backbone torsions.")
        else:
            raise ValueError(f"Invalid feature_type specified: {feature_type}")
        # --- End feature selection ---

        # 3. Load Trajectory
        print("Loading full trajectory...")
        traj = md.load(pdb_file, top=topology_mdtraj)
        print(f"Loaded trajectory with {traj.n_frames} frames.")
        if traj.n_frames == 0:
            raise ValueError("Trajectory contains no frames.")

        # 4. Alignment (if needed)
        if needs_alignment:
            heavy_atom_indices = topology_mdtraj.select('not element H and protein')
            if traj.n_frames < 1:
                 print("Warning: Trajectory has 0 frames, cannot align.", file=sys.stderr)
            elif reference_traj_for_alignment is None:
                 print("Warning: reference_traj_for_alignment not provided for heavy_atom features. Aligning to own first frame.", file=sys.stderr)
                 if traj.n_frames >= 2:
                     traj.superpose(traj, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=heavy_atom_indices)
            else:
                 print(f"Aligning trajectory to the provided reference structure (using heavy atoms)...")
                 # Align to the first frame of the reference trajectory
                 # Ensure heavy atom indices match the reference topology if it's different (though usually it's the same protein)
                 ref_heavy_atom_indices = reference_traj_for_alignment.topology.select('not element H and protein')
                 if len(heavy_atom_indices) != len(ref_heavy_atom_indices):
                     print("Warning: Number of heavy atoms differs between trajectory and reference. Alignment might be problematic.", file=sys.stderr)
                 traj.superpose(reference_traj_for_alignment, frame=0, atom_indices=heavy_atom_indices, ref_atom_indices=ref_heavy_atom_indices)


        # 5. Feature Calculation
        print(f"Calculating features ({feature_type})...")
        features_data = feat.transform(traj)

        if features_data is None or features_data.shape[0] == 0:
             raise RuntimeError("Feature data was not generated correctly.")

        print(f"Featurization successful. Feature shape: {features_data.shape}")

    except FileNotFoundError: pass
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
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Project SMD and ProToken trajectories onto a pre-computed TICA model and plot comparison.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'tica_model_file',
        help='Path to the saved TICA object file (*.pyemma) generated by pdb-analyze_tica.py.'
    )
    parser.add_argument(
        'smd_pdb_file',
        help='Path to the original SMD trajectory PDB file (used to train the TICA model).'
    )
    parser.add_argument(
        'protoken_pdb_file',
        help='Path to the ProToken trajectory PDB file to project and compare.'
    )
    # Feature type is now REQUIRED because it MUST match the loaded TICA model
    parser.add_argument(
        '-f', '--feature_type', # Added short flag
        choices=['ca_dist', 'heavy_atom', 'backbone_torsions'],
        required=True,
        help='Type of features used to build the TICA model. MUST match the features used to create the tica_model_file.'
    )
    parser.add_argument(
        '-o', '--output_plot',
        default=None,
        help='Output filename for the comparison plot. Defaults to TICA model dir + "comparison_<basename>.png".'
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
        # Make filename more descriptive based on TICA model name
        base_name = os.path.splitext(os.path.basename(args.tica_model_file))[0].replace('_model', '')
        args.output_plot = os.path.join(output_dir, f'comparison_{base_name}.png')
        print(f"No output plot path specified. Plot will be saved to: {args.output_plot}")
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    try:
        # 1. Load the saved TICA model
        print(f"Loading TICA model from: {args.tica_model_file}")
        tica_transformer = pyemma.load(args.tica_model_file)

        # --- Verify loaded object is indeed TICA ---
        if not isinstance(tica_transformer, coor.transform.TICA):
            raise TypeError(f"Loaded object from '{args.tica_model_file}' is not a PyEMMA TICA object (type: {type(tica_transformer)}).")
        # --- Extract necessary info from the loaded model (using API attributes) ---
        lag_time = tica_transformer.lag
        n_tica_dims = tica_transformer.dimension() # Output dimension

        # Get expected input feature dimension from the mean vector shape
        if hasattr(tica_transformer, 'mean') and tica_transformer.mean is not None:
            expected_feature_dim = tica_transformer.mean.shape[0]
        else:
            # Fallback or error if mean is not available (should be rare for a fitted TICA)
            print("Warning: Could not determine expected feature dimension from loaded TICA model's mean vector.", file=sys.stderr)
            # Attempt to get from cov if available, otherwise fail
            if hasattr(tica_transformer, 'cov') and tica_transformer.cov is not None:
                 expected_feature_dim = tica_transformer.cov.shape[0]
                 print(f"Using covariance matrix shape as fallback: {expected_feature_dim}", file=sys.stderr)
            else:
                 raise ValueError("Cannot determine expected input feature dimension from the loaded TICA model.")

        print(f"TICA model loaded: Lag={lag_time}, Expected Feature Dim={expected_feature_dim}, Output Dims={n_tica_dims}")

        if n_tica_dims < 1:
             raise ValueError("Loaded TICA model has 0 output dimensions.")

        # 2. Load reference structure for potential alignment (first frame of SMD)
        ref_traj_for_alignment = None
        if args.feature_type == 'heavy_atom':
            print("Loading first frame of SMD trajectory as alignment reference...")
            # Ensure topology is loaded correctly if needed for alignment atom selection
            ref_traj_for_alignment = md.load(args.smd_pdb_file, frame=0) # Load only frame 0
            if ref_traj_for_alignment.n_frames == 0:
                raise ValueError(f"Could not load frame 0 from SMD file '{args.smd_pdb_file}' for alignment reference.")


        # 3. Featurize and Project SMD Trajectory
        smd_features = load_trajectory_and_featurize(args.smd_pdb_file, args.feature_type, ref_traj_for_alignment)
        if smd_features is None:
            raise RuntimeError(f"Failed to featurize SMD trajectory: {args.smd_pdb_file}")

        # **Critical Check:** Verify feature dimension consistency
        if smd_features.shape[1] != expected_feature_dim:
              raise ValueError(f"Feature dimension mismatch for SMD! TICA model expects {expected_feature_dim} features, "
                               f"but SMD featurization yielded {smd_features.shape[1]}. "
                               f"Ensure --feature_type ('{args.feature_type}') matches the model and PDB files are compatible.")

        print("Projecting SMD features onto loaded TICA model...")
        Y_smd = tica_transformer.transform(smd_features)
        print(f"SMD projection successful. Shape: {Y_smd.shape}")


        # 4. Featurize and Project ProToken Trajectory
        pt_features = load_trajectory_and_featurize(args.protoken_pdb_file, args.feature_type, ref_traj_for_alignment)
        if pt_features is None:
            raise RuntimeError(f"Failed to featurize ProToken trajectory: {args.protoken_pdb_file}")

        # **Critical Check:** Verify feature dimension consistency
        if pt_features.shape[1] != expected_feature_dim:
             raise ValueError(f"Feature dimension mismatch for ProToken! TICA model expects {expected_feature_dim} features, "
                              f"but ProToken featurization yielded {pt_features.shape[1]}. "
                              f"Ensure --feature_type ('{args.feature_type}') matches the model and PDB files are compatible.")

        print("Projecting ProToken features onto loaded TICA model...")
        Y_pt = tica_transformer.transform(pt_features)
        print(f"ProToken projection successful. Shape: {Y_pt.shape}")


        # 5. Plotting Comparison
        print(f"Generating comparison plot...")
        fig, ax = plt.subplots(figsize=(10, 8))

        plot_dim = min(2, n_tica_dims) # Decide plot dimensionality based on TICA output

        if plot_dim >= 2: # Plot IC1 vs IC2
            xlabel = 'TICA Component 1 (IC1)'
            ylabel = 'TICA Component 2 (IC2)'
            plot_title = f'TICA Comparison (SMD vs ProToken, Lag={lag_time}, Feat={args.feature_type})'

            # Plot SMD trajectory as background scatter points
            # Slightly increase alpha for better visibility if needed, but keep it subtle
            ax.scatter(Y_smd[:, 0], Y_smd[:, 1], alpha=0.2, s=12, color="grey", label='SMD Projection', zorder=1)

            # Plot ProToken trajectory points, colored by frame index
            protoken_frames = np.arange(len(Y_pt))
            path_scatter = ax.scatter(Y_pt[:, 0], Y_pt[:, 1], c=protoken_frames, cmap='viridis', s=30, label='ProToken Projection (colored by frame)', zorder=3, edgecolors='grey', linewidth=0.5)

            # Add a *subtle* line connecting the points for path visualization
            ax.plot(Y_pt[:, 0], Y_pt[:, 1], color='black', alpha=0.3, linewidth=0.7, zorder=2)

            # Add colorbar for ProToken frame index
            cbar = fig.colorbar(path_scatter, ax=ax)
            cbar.set_label('ProToken Frame Index')

            # Add start/end markers for ProToken (make them stand out)
            ax.scatter(Y_pt[0, 0], Y_pt[0, 1], marker='^', s=180, color='lime', label='ProToken Start', zorder=4, edgecolors='black', linewidth=1)
            ax.scatter(Y_pt[-1, 0], Y_pt[-1, 1], marker='s', s=180, color='red', label='ProToken End', zorder=4, edgecolors='black', linewidth=1)

        elif plot_dim == 1: # Plot IC1 vs Frame Index
             xlabel = 'Frame Index'
             ylabel = 'TICA Component 1 (IC1)'
             plot_title = f'TICA Comparison - IC1 (Lag={lag_time}, Feat={args.feature_type})'

             # Plot SMD IC1 vs its frame index
             ax.plot(np.arange(len(Y_smd)), Y_smd[:, 0], alpha=0.5, color="grey", label='SMD IC1', linewidth=1.5)
             # Plot ProToken IC1 vs its frame index
             ax.plot(np.arange(len(Y_pt)), Y_pt[:, 0], marker='o', markersize=5, linestyle='-', color='blue', label='ProToken IC1') # Changed color

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(plot_title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to '{args.output_plot}'")
        plt.close(fig) # Close figure to release memory

    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}", file=sys.stderr)
        sys.exit(1)
    except (TypeError, ValueError, RuntimeError, IndexError) as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        # Uncomment traceback for detailed debugging if needed
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
