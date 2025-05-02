#!/usr/bin/env python3

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import distance_array
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from dtaidistance import dtw_ndim, dtw
import warnings
import argparse
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Helper Functions ---
def make_output_dir(dir_path):
    """Creates the output directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)

def save_figure(fig, output_path, verbose=True):
    """Saves the matplotlib figure."""
    try:
        fig.savefig(output_path, bbox_inches='tight')
        if verbose:
            print(f"Saved figure: {output_path}")
        plt.close(fig) # Close figure to free memory
    except Exception as e:
        print(f"Warning: Failed to save figure {output_path}. Error: {e}")
# --- End Helper Functions ---

def load_structure(pdb_path, selection="name CA", verbose=True):
    """Loads a single PDB structure and selects CA atoms."""
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if verbose:
        print(f"Loading structure: {pdb_path}")
    try:
        u = mda.Universe(pdb_path)
        atoms = u.select_atoms(selection)
        if len(atoms) == 0:
            raise ValueError(f"Selection '{selection}' yielded 0 atoms in {pdb_path}")
        if verbose:
            print(f"  Selected {len(atoms)} atoms using '{selection}'")
        return u, atoms
    except Exception as e:
        print(f"Error loading structure {pdb_path}: {e}")
        raise

def calculate_ca_distance_map(atoms):
    """Calculates the C-alpha distance map matrix."""
    coords = atoms.positions
    dist_vec = pdist(coords)
    dist_map = squareform(dist_vec)
    return dist_map

def plot_distance_map(dist_map, title, output_path, verbose=True):
    """Plots a distance map."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_map, cmap='viridis', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")
    plt.colorbar(im, label='Cα-Cα Distance (Å)', ax=ax)
    save_figure(fig, output_path, verbose)
# --- End Static Structure Functions ---


# --- MODIFIED: find_changing_pairs function ---
def find_changing_pairs(
    dist_map_start, dist_map_end, atoms_start,
    min_dist_change=1.0, top_percentage=0.2, # Changed from elbow_sensitivity to top_percentage
    verbose=True, output_dir="." # output_dir is kept for potential future plots, though elbow plot is removed
):
    """
    Finds Cα pairs with significant distance change, selecting the top percentage.

    Args:
        dist_map_start (np.ndarray): Distance map of start structure.
        dist_map_end (np.ndarray): Distance map of end structure.
        atoms_start (mda.AtomGroup): Atom group from start structure (for residue info).
        min_dist_change (float): Minimum absolute distance change to consider *before* selection.
        top_percentage (float): Percentage (0.0 to 1.0) of top changing pairs to select after filtering.
        verbose (bool): Print progress.
        output_dir (str): Directory for outputs (currently unused in this function).

    Returns:
        list: List of tuples `(idx1, idx2)` representing the selected atom indices.
        np.ndarray: Absolute distance difference map.
        list: List of tuples `(resname1, resid1, resname2, resid2, change)` for selected pairs.
    """
    if dist_map_start.shape != dist_map_end.shape:
        raise ValueError("Start and end distance maps have different shapes.")
    if not 0.0 < top_percentage <= 1.0:
        raise ValueError("top_percentage must be between 0.0 (exclusive) and 1.0 (inclusive).")

    dist_diff = np.abs(dist_map_end - dist_map_start)
    n_atoms = dist_diff.shape[0]

    # Get indices and values of upper triangle, excluding diagonal
    upper_tri_indices = np.triu_indices(n_atoms, k=1)
    all_diff_values = dist_diff[upper_tri_indices]
    all_atom_indices1 = upper_tri_indices[0]
    all_atom_indices2 = upper_tri_indices[1]

    # 1. Apply minimum distance change filter first
    valid_mask = all_diff_values >= min_dist_change
    if not np.any(valid_mask):
        raise ValueError(f"No Cα pairs found with distance change >= {min_dist_change} Å.")

    filtered_diffs = all_diff_values[valid_mask]
    filtered_indices1 = all_atom_indices1[valid_mask]
    filtered_indices2 = all_atom_indices2[valid_mask]

    # 2. Sort the filtered differences in descending order
    sort_order = np.argsort(filtered_diffs)[::-1]
    sorted_diffs = filtered_diffs[sort_order]
    sorted_indices1 = filtered_indices1[sort_order]
    sorted_indices2 = filtered_indices2[sort_order]

    # 3. Determine the number of pairs to select based on top_percentage
    num_filtered_pairs = len(sorted_diffs)
    num_selected = int(np.ceil(num_filtered_pairs * top_percentage))
    # Ensure at least one pair is selected if possible
    num_selected = max(1, num_selected)
    # Ensure we don't select more pairs than available
    num_selected = min(num_selected, num_filtered_pairs)


    selected_pairs_indices = []
    selected_pairs_info = []
    if verbose:
        print(f"Filtered {num_filtered_pairs} pairs with change >= {min_dist_change} Å.")
        print(f"Selecting top {num_selected} pairs ({top_percentage*100:.1f}% of filtered).")

    for i in range(num_selected):
        idx1 = sorted_indices1[i]
        idx2 = sorted_indices2[i]
        change = sorted_diffs[i]
        selected_pairs_indices.append((idx1, idx2))
        # Get residue info
        res1 = atoms_start[idx1].resname
        resid1 = atoms_start[idx1].resid
        res2 = atoms_start[idx2].resname
        resid2 = atoms_start[idx2].resid
        selected_pairs_info.append((res1, resid1, res2, resid2, change))

    if not selected_pairs_indices:
         # This should ideally not happen if the initial check passed, but as a safeguard:
         raise ValueError("Selection resulted in zero selected pairs after filtering.")

    return selected_pairs_indices, dist_diff, selected_pairs_info
# --- End MODIFIED function ---

def plot_difference_map(dist_diff, top_pairs_indices, title, output_path, verbose=True):
    """Plots the distance difference map, highlighting the selected pairs."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_diff, cmap='Reds', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")
    plt.colorbar(im, label='Absolute Distance Change (Å)', ax=ax)
    if top_pairs_indices and len(top_pairs_indices) < 500:
         xs = [p[1] for p in top_pairs_indices]
         ys = [p[0] for p in top_pairs_indices]
         ax.scatter(xs, ys, s=5, c='blue', alpha=0.5, label=f'Top {len(top_pairs_indices)} Changing Pairs')
         ax.scatter(ys, xs, s=5, c='blue', alpha=0.5)
    save_figure(fig, output_path, verbose)

def save_selected_pairs(pairs_indices, pairs_info, output_dir, verbose=True):
    """Saves selected pairs (indices and readable info)."""
    indices_path = os.path.join(output_dir, "selected_ca_pairs_indices.pkl")
    info_path = os.path.join(output_dir, "selected_ca_pairs_info.txt")
    with open(indices_path, 'wb') as f:
        pickle.dump(pairs_indices, f)
    if verbose:
        print(f"Saved selected pair indices to: {indices_path}")
    with open(info_path, 'w') as f:
        # Update comment to reflect the selection method
        f.write(f"# Selected C-alpha Pairs (Top {args.top_percentage*100:.1f}% changing pairs, min_change={args.min_dist_change}A)\n")
        f.write("# Res1, ResID1, Res2, ResID2, AbsDistChange_A\n")
        for info in pairs_info:
            f.write(f"{info[0]}, {info[1]}, {info[2]}, {info[3]}, {info[4]:.3f}\n")
    if verbose:
        print(f"Saved selected pair readable info to: {info_path}")

def load_trajectory(traj_path, top_path=None, verbose=True):
    """
    Loads trajectory. If traj_path is a PDB, it's used for both topology and coordinates.
    Otherwise, top_path is required.
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    is_pdb = traj_path.lower().endswith(('.pdb', '.ent'))

    try:
        if is_pdb:
            if top_path and verbose:
                print(f"  Info: Trajectory '{os.path.basename(traj_path)}' is a PDB. Ignoring provided topology '{os.path.basename(top_path)}'.")
            if verbose:
                print(f"Loading PDB Trajectory (topology & coords): '{traj_path}'")
            print("  Loading PDB structure and coordinates into memory...")
            # For PDB, topology and coordinates are in the same file
            u = mda.Universe(traj_path, in_memory=True)
            num_atoms = len(u.atoms)
            if verbose:
                print(f"  PDB '{os.path.basename(traj_path)}' loaded with {num_atoms} atoms.")

        else: # Not a PDB, requires separate topology
            if not top_path:
                raise ValueError(f"Trajectory file '{traj_path}' is not a PDB and requires a topology file (--topology1/--topology2).")
            if not os.path.exists(top_path):
                raise FileNotFoundError(f"Provided topology file not found for {traj_path}: {top_path}")
            if verbose:
                print(f"Loading Trajectory: Topology='{top_path}', Trajectory='{traj_path}'")

            print("  Loading trajectory structure (topology) into memory...")
            # Load topology first to get expected atom count (optional, but good for error check)
            structure_u = mda.Universe(top_path)
            num_top_atoms = len(structure_u.atoms)
            if verbose:
                print(f"  Topology '{os.path.basename(top_path)}' expects {num_top_atoms} atoms.")

            print("  Loading trajectory coordinates into memory...")
            # Load the trajectory, associating it with the topology
            u = mda.Universe(top_path, traj_path, in_memory=True)
            num_atoms = len(u.atoms)

            # Check atom count consistency
            if len(u.trajectory) > 0 and u.atoms.n_atoms != num_top_atoms:
                 raise ValueError(f"Atom count mismatch in {traj_path}: Topology has {num_top_atoms}, "
                                  f"but trajectory frame reports {u.atoms.n_atoms}.")

        if verbose:
            print(f"Successfully loaded Trajectory ({num_atoms} atoms, {len(u.trajectory)} frames)")
        return u

    except IndexError as ie:
        # Catch the specific IndexError potentially caused by topology/trajectory mismatch (less likely for PDB-only load)
        print(f"\nError: IndexError encountered while reading trajectory '{os.path.basename(traj_path)}'.")
        if not is_pdb and top_path:
             print(f"This often indicates an inconsistency between the topology file '{os.path.basename(top_path)}' and the trajectory file '{os.path.basename(traj_path)}'.")
        elif is_pdb:
             print(f"This might indicate issues with the PDB file format or content in '{os.path.basename(traj_path)}'.")
        else: # Should not happen based on logic, but as fallback
             print("An unexpected indexing error occurred during file loading.")
        print(f"Original error: {ie}")
        print("Please check the file formats and consistency.")
        raise # Re-raise the exception
    except MemoryError:
        print(f"MemoryError loading {traj_path}. Trajectory too large for memory.")
        raise
    except Exception as e:
        print(f"Error loading trajectory {traj_path}: {e}")
        raise

def extract_trajectory_features(universe, pairs_indices, selection="name CA"):
    """Extracts distances for selected Cα pairs over a trajectory."""
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_pairs = len(pairs_indices)
    if n_pairs == 0:
        raise ValueError("Cannot extract features: No pairs were selected.")
    features = np.empty((n_frames, n_pairs), dtype=np.float32)
    pair_indices_array = np.array(pairs_indices, dtype=int)
    # Check if any pair index is out of bounds for the selected atoms
    max_index_required = pair_indices_array.max()
    if max_index_required >= len(atoms):
         raise IndexError(f"Selected pair index {max_index_required} is out of bounds "
                          f"for the selected atoms group (size {len(atoms)}). "
                          "Ensure start/end PDBs and trajectories are consistent.")

    for i, ts in enumerate(universe.trajectory):
        coords = atoms.positions
        distances = distance_array(coords, coords, box=ts.dimensions)[pair_indices_array[:, 0], pair_indices_array[:, 1]]
        features[i, :] = distances
    return features

def extract_static_features(atoms, pairs_indices):
    """Extracts distances for selected Cα pairs for a static structure."""
    n_pairs = len(pairs_indices)
    if n_pairs == 0:
        raise ValueError("Cannot extract features: No pairs were selected.")
    features = np.empty((1, n_pairs), dtype=np.float32)
    pair_indices_array = np.array(pairs_indices, dtype=int)
    max_index_required = pair_indices_array.max()
    if max_index_required >= len(atoms):
         raise IndexError(f"Selected pair index {max_index_required} is out of bounds "
                          f"for the static atoms group (size {len(atoms)}).")
    coords = atoms.positions
    distances = distance_array(coords, coords)[pair_indices_array[:, 0], pair_indices_array[:, 1]]
    features[0, :] = distances
    return features


# ... (Keep perform_pca_and_transform, save_pca_model, plot_projected_paths, calculate_and_plot_dtw) ...
def perform_pca_and_transform(features1, features2, features_start, features_end, n_components=3, verbose=True):
    """Fits PCA on features1 and transforms all feature sets."""
    if verbose:
        print(f"\nFitting PCA on Traj1 (SMD) features (shape {features1.shape}) using {n_components} components...")

    combined_features = np.vstack((features1, features2)) # Combine for potentially better PCA space

    pca = PCA(n_components=n_components)
    # pca.fit(features1) # Option: Fit only on traj1
    pca.fit(combined_features) # Option: Fit on combined data
    if verbose:
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        # print(f"PCA fitted (on Traj1). Explained variance: {explained_variance:.2f}%")
        print(f"PCA fitted (on combined data). Explained variance: {explained_variance:.2f}%")


    if verbose: print("Transforming all features to PCA space...")
    proj1 = pca.transform(features1)
    proj2 = pca.transform(features2)
    proj_start1 = pca.transform(features_start)
    proj_end1 = pca.transform(features_end)

    if verbose:
        print(f"Projected shapes: Traj1={proj1.shape}, Traj2={proj2.shape}, Start={proj_start1.shape}, End={proj_end1.shape}")

    return pca, proj1, proj2, proj_start1, proj_end1

def save_pca_model(pca_model, output_dir, verbose=True):
    """Saves the fitted PCA model using pickle."""
    pca_path = os.path.join(output_dir, "pca_model.pkl")
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_model, f)
    if verbose:
        print(f"Saved PCA model to: {pca_path}")

def plot_projected_paths(proj1, proj2, proj_start, proj_end, traj1_name, traj2_name, output_path, verbose=True):
    """Plots the projected trajectories in the first 2 PCA dimensions."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(proj1[:, 0], proj1[:, 1], 'o-', label=f'Traj 1 ({traj1_name})', color='C0', alpha=0.7, markersize=3, linewidth=1, zorder=2)
    ax.plot(proj2[:, 0], proj2[:, 1], 's-', label=f'Traj 2 ({traj2_name})', color='C1', alpha=0.7, markersize=3, linewidth=1, zorder=3)
    ax.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='Traj 1 Start', zorder=4)
    ax.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='Traj 1 End', zorder=4)
    ax.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='Traj 2 Start', zorder=5)
    ax.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='Traj 2 End', zorder=5)
    ax.plot(proj_start[:, 0], proj_start[:, 1], 'P', color='green', markersize=12, label='Ref Start PDB', zorder=10, mec='black')
    ax.plot(proj_end[:, 0], proj_end[:, 1], 'X', color='red', markersize=12, label='Ref End PDB', zorder=10, mec='black')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize='small', loc='best')
    ax.set_title("Projected Trajectories in PCA Space (PC1 vs PC2)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    save_figure(fig, output_path, verbose)

def calculate_and_plot_dtw(proj1, proj2, traj1_name, traj2_name, output_path, verbose=True):
    """Calculates DTW and plots the alignment visualization."""
    if verbose:
        print("\nCalculating DTW distance on PCA components...")
    proj1_dtw = np.ascontiguousarray(proj1, dtype=np.double)
    proj2_dtw = np.ascontiguousarray(proj2, dtype=np.double)
    window_arg = max(10, abs(len(proj1_dtw) - len(proj2_dtw)))
    if verbose: print(f"  Using DTW window size: {window_arg}")
    try:
        distance, paths = dtw_ndim.warping_paths(proj1_dtw, proj2_dtw, window=window_arg)
    except Exception as e:
         print(f"Warning: DTW calculation with window failed: {e}. Trying without window...")
         try:
              distance, paths = dtw_ndim.warping_paths(proj1_dtw, proj2_dtw)
         except Exception as e2:
              print(f"Error: DTW failed even without windowing: {e2}")
              return np.inf, None
    if verbose:
        print(f"DTW calculation complete. Distance = {distance:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(f"DTW Path Comparison (PCA Space, Dist={distance:.2f})", fontsize=14)
    ax1 = axs[0]
    ax1.plot(proj1[:, 0], proj1[:, 1], 'o-', label=f'Traj 1 ({traj1_name})', color='C0', alpha=0.7, markersize=3, linewidth=1)
    ax1.plot(proj2[:, 0], proj2[:, 1], 's-', label=f'Traj 2 ({traj2_name})', color='C1', alpha=0.7, markersize=3, linewidth=1)
    ax1.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='Traj 1 Start')
    ax1.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='Traj 1 End')
    ax1.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='Traj 2 Start')
    ax1.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='Traj 2 End')
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.legend(fontsize='small')
    ax1.set_title("Paths in PCA Space (PC1 vs PC2)")
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axs[1]
    try:
        best_path = dtw.best_path(paths)
    except Exception as path_err:
         print(f"Warning: Could not extract best path from DTW result: {path_err}")
         best_path = []
    ax2.plot(proj1[:, 0], proj1[:, 1], '-', color='C0', alpha=0.3, linewidth=1)
    ax2.plot(proj2[:, 0], proj2[:, 1], '-', color='C1', alpha=0.3, linewidth=1)
    if best_path:
        for idx1, idx2 in best_path:
            if idx1 < len(proj1) and idx2 < len(proj2):
                 ax2.plot([proj1[idx1, 0], proj2[idx2, 0]],
                          [proj1[idx1, 1], proj2[idx2, 1]],
                          '-', color='gray', linewidth=0.5, alpha=0.4)
    ax2.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8)
    ax2.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8)
    ax2.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8)
    ax2.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8)
    ax2.set_xlabel("PC 1")
    ax2.set_title("DTW Alignment Path")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, output_path, verbose)
    return distance, paths
# --- End Analysis and Plotting Functions ---



# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two trajectories using PCA on changing Cα-Cα distances and DTW. Selects top % changing pairs." # Modified description
    )
    # Input Files
    parser.add_argument("start_pdb", help="Path to the starting structure PDB file.")
    parser.add_argument("end_pdb", help="Path to the ending structure PDB file.")
    parser.add_argument("traj1", help="Path to the first trajectory file (e.g., SMD PDB or other format).")
    parser.add_argument("traj2", help="Path to the second trajectory file (e.g., ProToken PDB or other format).")
    parser.add_argument("--topology1", default=None, help="Optional topology for traj1 (required if traj1 is not PDB).")
    parser.add_argument("--topology2", default=None, help="Optional topology for traj2 (required if traj2 is not PDB).")
    # Parameters
    parser.add_argument("-s", "--atom_selection", default="protein and name CA", help="Atom selection for Cα atoms.")
    parser.add_argument("-n", "--n_components", type=int, default=3, help="Number of PCA components.")
    # --- Parameters for automatic pair selection ---
    parser.add_argument("--min_dist_change", type=float, default=1.0, help="Minimum absolute distance change (Å) to consider a pair *before* selection.")
    # Removed elbow_sensitivity, added top_percentage
    parser.add_argument("--top_percentage", type=float, default=0.2, help="Percentage (0.0 to 1.0) of top changing pairs to select (default: 0.2 = 20%).")
    # Output
    parser.add_argument("-o", "--output_dir", default="dtw_pca_dist_analysis", help="Directory to save outputs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if args.n_components < 2:
         print("Warning: Plotting requires at least 2 PCA components. Setting n_components to 2 for plotting.")
         args.n_components = max(args.n_components, 2)

    # Validate top_percentage
    if not 0.0 < args.top_percentage <= 1.0:
        parser.error("--top_percentage must be between 0.0 (exclusive) and 1.0 (inclusive).")

    # Make output dir absolute path upfront
    args.output_dir = os.path.abspath(args.output_dir)


    try:
        # --- 0. Setup Output Directory ---
        make_output_dir(args.output_dir)
        if args.verbose:
            print(f"Outputs will be saved to: {args.output_dir}")

        # --- 1. Load Static Structures & Calculate Initial/Final Distance Maps ---
        ref_start_u, ref_start_atoms = load_structure(args.start_pdb, args.atom_selection, args.verbose)
        ref_end_u, ref_end_atoms = load_structure(args.end_pdb, args.atom_selection, args.verbose)
        if len(ref_start_atoms) != len(ref_end_atoms):
            raise ValueError("Start and End PDBs have different numbers of selected atoms!")
        dist_map_start = calculate_ca_distance_map(ref_start_atoms)
        dist_map_end = calculate_ca_distance_map(ref_end_atoms)
        plot_distance_map(dist_map_start, "Start Structure Cα Distances", os.path.join(args.output_dir, "distance_map_start.png"), args.verbose)
        plot_distance_map(dist_map_end, "End Structure Cα Distances", os.path.join(args.output_dir, "distance_map_end.png"), args.verbose)

        # --- 2. Find and Save Top Changing Pairs using TOP PERCENTAGE method ---
        selected_pairs_indices, dist_diff, selected_pairs_info = find_changing_pairs(
            dist_map_start, dist_map_end, ref_start_atoms,
            args.min_dist_change, args.top_percentage, args.verbose, args.output_dir # Pass top_percentage
        )
        if args.verbose:
            print(f"\nSelected {len(selected_pairs_indices)} Cα pairs (Top {args.top_percentage*100:.1f}%).")

        plot_difference_map(dist_diff, selected_pairs_indices, f"Absolute Distance Change ({len(selected_pairs_indices)} pairs selected)",
                             os.path.join(args.output_dir, "distance_map_difference.png"), args.verbose)
        save_selected_pairs(selected_pairs_indices, selected_pairs_info, args.output_dir, args.verbose)

        # --- 3. Load Trajectories ---
        # Pass topology explicitly only if provided
        u1 = load_trajectory(args.traj1, args.topology1, args.verbose)
        u2 = load_trajectory(args.traj2, args.topology2, args.verbose)

        # --- Consistency Check ---
        traj1_atoms_check = u1.select_atoms(args.atom_selection)
        traj2_atoms_check = u2.select_atoms(args.atom_selection)
        if len(traj1_atoms_check) != len(ref_start_atoms):
            raise ValueError(f"Traj1 selected atom count ({len(traj1_atoms_check)}) differs from static ref ({len(ref_start_atoms)}).")
        if len(traj2_atoms_check) != len(ref_start_atoms):
             raise ValueError(f"Traj2 selected atom count ({len(traj2_atoms_check)}) differs from static ref ({len(ref_start_atoms)}).")

        # --- 4. Extract Features for Trajectories and Static Points ---
        if args.verbose: print("\nExtracting features (selected distances) for trajectories...")
        features1 = extract_trajectory_features(u1, selected_pairs_indices, args.atom_selection)
        features2 = extract_trajectory_features(u2, selected_pairs_indices, args.atom_selection)
        if args.verbose: print("Extracting features for static start/end points...")
        features_start = extract_static_features(ref_start_atoms, selected_pairs_indices)
        features_end = extract_static_features(ref_end_atoms, selected_pairs_indices)

        # --- 5. Perform PCA and Transform ---
        pca_model, proj1, proj2, proj_start, proj_end = perform_pca_and_transform(
            features1, features2, features_start, features_end, args.n_components, args.verbose
        )
        save_pca_model(pca_model, args.output_dir, args.verbose)

        # --- 6. Plot Projected Paths ---
        plot_projected_paths(
            proj1, proj2, proj_start, proj_end,
            os.path.basename(args.traj1), os.path.basename(args.traj2),
            os.path.join(args.output_dir, "projected_paths_overlay.png"), args.verbose
        )

        # --- 7. Calculate DTW and Plot Alignment ---
        dtw_distance, _ = calculate_and_plot_dtw(
            proj1, proj2,
            os.path.basename(args.traj1), os.path.basename(args.traj2),
            os.path.join(args.output_dir, "dtw_alignment.png"), args.verbose
        )
        if np.isfinite(dtw_distance):
            print(f"\nFinal DTW Distance based on {len(selected_pairs_indices)} automatically selected Cα distances: {dtw_distance:.4f}")
        else:
            print("\nDTW calculation failed or resulted in infinity.")


    # ... (Error handling remains the same) ...
    except FileNotFoundError as fnf_err:
         print(f"\nError: {fnf_err}")
    except ValueError as ve:
        print(f"\nValueError during processing: {ve}")
        import traceback
        traceback.print_exc()
    except MemoryError:
         print("\nMemoryError encountered, likely during processing.")
    except ImportError as imp_err:
        print(f"\nImportError: {imp_err}. Required libraries not found.")
        print("Please install: pip install MDAnalysis dtaidistance numpy scikit-learn matplotlib scipy")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()

    print("\nAnalysis complete.")
