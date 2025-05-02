#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.analysis import align, pca # Import align and pca
import numpy as np
from dtaidistance import dtw_ndim # Use the ndim version
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
from sklearn.decomposition import PCA # Import PCA
import warnings
import argparse
import os
import matplotlib.pyplot as plt # Moved import here

# Suppress specific MDAnalysis warnings if they become too verbose
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.analysis.rms')
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis.analysis.align')


def load_universe(traj_path, top_path=None, verbose=True):
    """
    Loads an MDAnalysis Universe, automatically using the trajectory
    as topology if it's a PDB file and no topology is provided.
    (Code identical to previous version)
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    is_pdb = traj_path.lower().endswith(('.pdb', '.ent'))
    final_top_path = top_path

    if final_top_path:
        if not os.path.exists(final_top_path):
            raise FileNotFoundError(f"Provided topology file not found: {final_top_path}")
        if verbose:
            print(f"Loading Universe: Topology='{final_top_path}', Trajectory='{traj_path}'")
    elif is_pdb:
        final_top_path = traj_path
        if verbose:
            print(f"Loading Universe: Using PDB trajectory as topology and coordinates: '{traj_path}'")
    else:
        raise ValueError(f"Trajectory file '{traj_path}' is not a PDB format. "
                         "A separate topology file must be provided using --topology1/--topology2.")

    try:
        # Load into memory for easier coordinate access needed for PCA
        # If memory is an issue, this part needs chunking or a different approach.
        print("  Loading trajectory into memory for PCA preprocessing...")
        universe = mda.Universe(final_top_path, traj_path, in_memory=True)
        if verbose:
            print(f"Successfully loaded Universe ({len(universe.atoms)} atoms, {len(universe.trajectory)} frames)")
        return universe
    except MemoryError:
        print(f"MemoryError loading {traj_path}. Trajectory might be too large to load into memory for PCA.")
        raise
    except Exception as e:
        print(f"Error loading Universe with topology '{final_top_path}' and trajectory '{traj_path}':")
        raise e


def get_aligned_coords(universe, selection="protein and name CA", ref_universe=None, verbose=True):
    """
    Selects atoms, aligns the trajectory to a reference, and returns coordinates.

    Args:
        universe (mda.Universe): The universe to process.
        selection (str): Atom selection string.
        ref_universe (mda.Universe, optional): Reference universe for alignment.
                                               If None, aligns to the first frame of `universe`.
        verbose (bool): Print progress.

    Returns:
        np.ndarray: Aligned coordinates of shape (n_frames, n_atoms * 3).
        mda.AtomGroup: The selected atom group.
    """
    if verbose:
        print(f"Selecting atoms: '{selection}'")
    atomgroup = universe.select_atoms(selection)
    n_atoms = len(atomgroup)
    if n_atoms == 0:
        raise ValueError(f"Atom selection '{selection}' resulted in zero atoms.")

    if verbose:
        print(f"Aligning trajectory ({len(universe.trajectory)} frames, {n_atoms} atoms) to reference...")

    # Determine reference for alignment
    if ref_universe:
        ref_atoms = ref_universe.select_atoms(selection)
        if len(ref_atoms) != n_atoms:
             raise ValueError("Reference universe has different number of atoms for the same selection.")
        ref_coords = ref_atoms.positions # Use first frame of ref_universe
    else:
        # Align to the first frame of the current trajectory
        universe.trajectory[0] # Go to first frame
        ref_coords = atomgroup.positions.copy()

    # Perform alignment (modifies coordinates in the trajectory object)
    # Use strict=False to avoid errors if frames have slight atom count differences temporarily
    # (though the initial load should prevent this for the whole trajectory)
    align.AlignTraj(universe,                   # Universe to align
                    reference=ref_universe or universe,            # Reference universe (optional, coords used below)
                    select=selection,          # Selection string for alignment
                    filename=None,             # Don't write output file
                    prefix=None,
                    weights=None,              # Could use 'mass'
                    ref_frame=0,               # Reference frame index within reference if ref universe provided
                    in_memory=True,
                    match_atoms=False,         # Assume atom order is the same
                    reference_coords=ref_coords,# Provide explicit coords
                    verbose=False).run()      # Suppress AlignTraj verbosity

    if verbose:
         print("Alignment complete. Extracting coordinates...")

    # Extract aligned coordinates and reshape
    n_frames = len(universe.trajectory)
    coords = np.empty((n_frames, n_atoms * 3), dtype=np.float32) # Use float32 for PCA
    for i, ts in enumerate(universe.trajectory):
        coords[i, :] = atomgroup.positions.reshape(-1) # Flatten coordinates

    if verbose:
         print(f"Extracted coordinates shape: {coords.shape}")

    return coords, atomgroup

def calculate_dtw_on_pca(
    u1, u2, atom_selection="protein and name CA", n_pca_components=3, verbose=True
):
    """
    Performs PCA on aligned coordinates and calculates DTW distance
    on the principal components.

    Args:
        u1 (MDAnalysis.Universe): First trajectory universe.
        u2 (MDAnalysis.Universe): Second trajectory universe.
        atom_selection (str): Common atom selection string.
        n_pca_components (int): Number of principal components to keep.
        verbose (bool): Print progress.

    Returns:
        float: The DTW distance.
        np.ndarray: Projected coordinates for trajectory 1.
        np.ndarray: Projected coordinates for trajectory 2.
        object: DTW path information from dtaidistance.
        PCA: PCA object used for projection.
    """
    # 1. Align trajectories and get coordinates
    # Align u2 to the first frame of u1 for a common reference frame
    if verbose: print("\n--- Processing Trajectory 1 ---")
    coords1, ag1 = get_aligned_coords(u1, atom_selection, ref_universe=None, verbose=verbose) # Align u1 to its own first frame

    if verbose: print("\n--- Processing Trajectory 2 ---")
    coords2, ag2 = get_aligned_coords(u2, atom_selection, ref_universe=u1, verbose=verbose) # Align u2 to first frame of u1

    # Check consistency one last time
    if len(ag1) != len(ag2):
         raise ValueError("Internal error: Atom selections yielded different atom counts after loading.")

    # 2. Fit PCA
    if verbose:
        print(f"\nFitting PCA on Trajectory 1 coordinates (keeping {n_pca_components} components)...")
    # It's generally better to fit PCA on the longer or more representative trajectory,
    # or potentially both combined (requires concatenation). Let's fit on u1 for now.
    pca = PCA(n_components=n_pca_components)
    pca.fit(coords1)
    if verbose:
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA fitted. Explained variance by {n_pca_components} components: {explained_variance:.2f}%")

    # 3. Transform both coordinate sets
    if verbose: print("Transforming coordinates to PCA space...")
    proj_coords1 = pca.transform(coords1)
    proj_coords2 = pca.transform(coords2)
    if verbose:
        print(f"Projected coordinates shape: Traj1={proj_coords1.shape}, Traj2={proj_coords2.shape}")

    # 4. Calculate DTW on projected coordinates using dtaidistance.dtw_ndim
    if verbose:
        print("\nCalculating DTW distance on PCA components...")

    # Ensure data is double precision for dtaidistance C library
    proj_coords1_dtw = np.ascontiguousarray(proj_coords1, dtype=np.double)
    proj_coords2_dtw = np.ascontiguousarray(proj_coords2, dtype=np.double)

    # Use dtw_ndim.warping_paths for distance and the path matrix
    # You might need to adjust window, max_dist etc. based on your data
    # distance = dtw_ndim.distance(proj_coords1_dtw, proj_coords2_dtw) # Just the distance
    distance, paths = dtw_ndim.warping_paths(proj_coords1_dtw, proj_coords2_dtw)


    if verbose:
        print(f"DTW calculation complete. Distance = {distance}")

    # Note: 'paths' here is the accumulated cost matrix from dtaidistance,
    # suitable for dtwvis.plot_warpingpaths
    return distance, proj_coords1, proj_coords2, paths, pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate DTW path similarity on PCA-projected coordinates. Automatically uses PDB trajectories as topologies if no specific topology file is given."
    )
    # Trajectory Arguments
    parser.add_argument("traj1", help="SMD's trajectory file path (e.g., PDB, DCD, XTC).")
    parser.add_argument("traj2", help="ProToken's trajectory file path (e.g., PDB, DCD, XTC).")
    # Optional Topology Arguments
    parser.add_argument("--topology1", default=None, help="Optional topology file for trajectory 1.")
    parser.add_argument("--topology2", default=None, help="Optional topology file for trajectory 2.")
    # Calculation Parameters
    parser.add_argument(
        "--atom-selection", default="protein and name CA",
        help="Atom selection string for alignment and PCA (MDAnalysis syntax)."
    )
    parser.add_argument(
        "-n", "--n-components", type=int, default=3,
        help="Number of principal components to use for DTW."
    )
    # Output Arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", "--output", default="dtw_pca_paths.png", help="Output filename for DTW path visualization")
    # New reference structure arguments
    parser.add_argument('-s', "--start-struct", default=None, help="Optional PDB for start reference")
    parser.add_argument('-e', "--end-struct", default=None, help="Optional PDB for end reference")

    args = parser.parse_args()

    try:
        # --- Load Universes (might load into memory) ---
        u1 = load_universe(args.traj1, args.topology1, args.verbose)
        u2 = load_universe(args.traj2, args.topology2, args.verbose)

        # --- Perform PCA and DTW ---
        dtw_distance, proj1, proj2, dtw_acc_cost_matrix, pca = calculate_dtw_on_pca(
            u1, u2,
            atom_selection=args.atom_selection,
            n_pca_components=args.n_components,
            verbose=args.verbose
        )

        # Load and project reference structures
        proj_start = proj_end = None
        if args.start_struct:
            if not os.path.exists(args.start_struct):
                raise FileNotFoundError(f"Start structure not found: {args.start_struct}")
            start_u = load_universe(args.start_struct, None, args.verbose)
            coords_s, _ = get_aligned_coords(start_u, args.atom_selection, ref_universe=u1, verbose=False)
            proj_start = pca.transform(coords_s)
        if args.end_struct:
            if not os.path.exists(args.end_struct):
                raise FileNotFoundError(f"End structure not found: {args.end_struct}")
            end_u = load_universe(args.end_struct, None, args.verbose)
            coords_e, _ = get_aligned_coords(end_u, args.atom_selection, ref_universe=u1, verbose=False)
            proj_end = pca.transform(coords_e)

        # --- Process and Plot Results ---
        if np.isfinite(dtw_distance):
            print(f"\nDTW Path Similarity Distance (on {args.n_components} PCs of '{args.atom_selection}'): {dtw_distance:.4f}")

            if proj1 is not None and proj2 is not None and dtw_acc_cost_matrix is not None:
                try:
                    # --- NEW Plotting Code ---
                    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True) # Slightly increased height for legend
                    fig.suptitle(f"DTW Path Comparison (PCA Space, Dist={dtw_distance:.2f})", fontsize=14)

                    # --- Plot 1: Path Overlay ---
                    ax1 = axs[0]
                    # Use shorter basenames for legend
                    basename1 = os.path.basename(args.traj1)
                    basename2 = os.path.basename(args.traj2)
                    label1 = f'Traj 1 ({basename1})'
                    label2 = f'Traj 2 ({basename2})'
                    max_label_len = 25 # Limit length if needed
                    if len(label1) > max_label_len: label1 = label1[:max_label_len-3] + '...'
                    if len(label2) > max_label_len: label2 = label2[:max_label_len-3] + '...'


                    # Plot paths and store handles/labels implicitly via labels
                    ax1.plot(proj1[:, 0], proj1[:, 1], 'o-', label=label1, alpha=0.7, markersize=3, linewidth=1)
                    ax1.plot(proj2[:, 0], proj2[:, 1], 's-', label=label2, alpha=0.7, markersize=3, linewidth=1)

                    # Annotate reference structures
                    if proj_start is not None:
                        ax1.plot(proj_start[0, 0], proj_start[0, 1], 'ko', markersize=8, label='Start Ref')
                    if proj_end is not None:
                        ax1.plot(proj_end[0, 0], proj_end[0, 1], 'k^', markersize=8, label='End Ref')

                    # Mark start and end points
                    ax1.plot(proj1[0, 0], proj1[0, 1], 'gx', markersize=8, label='Traj 1 Start')
                    ax1.plot(proj1[-1, 0], proj1[-1, 1], 'g+', markersize=8, label='Traj 1 End')
                    ax1.plot(proj2[0, 0], proj2[0, 1], 'mx', markersize=8, label='Traj 2 Start')
                    ax1.plot(proj2[-1, 0], proj2[-1, 1], 'm+', markersize=8, label='Traj 2 End')
                    ax1.set_xlabel("PC 1")
                    ax1.set_ylabel("PC 2")

                    # ax1.legend(fontsize='small', loc='upper left')
                    ax1.set_title("Paths in PCA Space (PC1 vs PC2)")
                    ax1.set_aspect('equal', adjustable='box')
                    ax1.grid(True, linestyle='--', alpha=0.6)

                    # --- Plot 2: DTW Alignment Visualization ---
                    ax2 = axs[1]
                    best_path = dtw.best_path(dtw_acc_cost_matrix) # Use dtaidistance helper

                    # Plot original paths lightly (no labels needed for figure legend)
                    ax2.plot(proj1[:, 0], proj1[:, 1], '-', color='C0', alpha=0.3, linewidth=1)
                    ax2.plot(proj2[:, 0], proj2[:, 1], '-', color='C1', alpha=0.3, linewidth=1)

                    # Draw lines connecting aligned points
                    for idx1, idx2 in best_path:
                        # Check bounds just in case (shouldn't be necessary if DTW worked)
                        if idx1 < len(proj1) and idx2 < len(proj2):
                             ax2.plot([proj1[idx1, 0], proj2[idx2, 0]],
                                      [proj1[idx1, 1], proj2[idx2, 1]],
                                      '-', color='gray', linewidth=0.5, alpha=0.4)

                    # Annotate reference structures (no labels needed)
                    if proj_start is not None:
                        ax2.plot(proj_start[0, 0], proj_start[0, 1], 'ko', markersize=8)
                    if proj_end is not None:
                        ax2.plot(proj_end[0, 0], proj_end[0, 1], 'k^', markersize=8)

                    # Re-plot start/end points on top for clarity (no labels needed)
                    ax2.plot(proj1[0, 0], proj1[0, 1], 'gx', markersize=8)
                    ax2.plot(proj1[-1, 0], proj1[-1, 1], 'g+', markersize=8)
                    ax2.plot(proj2[0, 0], proj2[0, 1], 'mx', markersize=8)
                    ax2.plot(proj2[-1, 0], proj2[-1, 1], 'm+', markersize=8)

                    ax2.set_xlabel("PC 1")
                    # ax2.set_ylabel("PC 2") # Y-axis is shared
                    ax2.set_title("DTW Alignment Path")
                    ax2.set_aspect('equal', adjustable='box')
                    ax2.grid(True, linestyle='--', alpha=0.6)

                    # --- Create Figure Legend Below Plots ---
                    handles, labels = ax1.get_legend_handles_labels() # Get everything from ax1
                    # Place legend below the subplots, centered, with 4 columns
                    fig.legend(handles, labels, fontsize='small', loc='upper left')

                    # Adjust layout to make room for the figure legend and title
                    # Increase bottom margin (rect[1]) from default ~0.1 to ~0.15 or more
                    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust bottom margin (0.1) if legend overlaps x-labels

                    plt.savefig(args.output, bbox_inches='tight') # Use bbox_inches='tight' for potentially better fitting
                    if args.verbose:
                        print(f"\nSaved DTW path visualization to {args.output}")

                except ImportError:
                    print("\nInstall matplotlib (pip install matplotlib) and scikit-learn (pip install scikit-learn) to visualize the results.")
                except Exception as plot_err:
                    print(f"\nError during plotting: {plot_err}")
                    import traceback
                    traceback.print_exc() # Print detailed plotting error
            else:
                 print("\nCalculation produced invalid results, skipping plot.")

    except FileNotFoundError as fnf_err:
         print(f"\nError: {fnf_err}")
    except ValueError as ve:
        print(f"\nValueError during processing: {ve}")
    except MemoryError:
         print("\nMemoryError encountered, likely during trajectory loading or PCA.")
    except ImportError as imp_err:
        print(f"\nImportError: {imp_err}. Required libraries not found.")
        print("Please install: pip install MDAnalysis dtaidistance numpy scikit-learn matplotlib")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()
