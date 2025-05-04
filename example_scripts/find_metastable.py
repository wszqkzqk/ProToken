#!/usr/bin/env python3

import MDAnalysis as mda
import numpy as np
from sklearn.decomposition import PCA # Only needed for type hints if strict
from dtaidistance import dtw_ndim, dtw
import warnings
import argparse
import os
import pickle
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform # Needed if recalculating anything distance related, keep for now
from MDAnalysis.lib.distances import distance_array # Efficient distance calculation


# --- Plotting Setup ---
import matplotlib
# Try Agg backend first for non-GUI environments
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Using Agg backend for Matplotlib.")
except ImportError:
    # Fallback if Agg is not available (less common)
    import matplotlib.pyplot as plt
    print("Agg backend not available, using default Matplotlib backend.")


# --- Constants ---
EPSILON = 1e-6 # To avoid division by zero

# --- Helper Functions ---
def make_output_dir(dir_path):
    """Creates the output directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)

def save_figure(fig, output_path, verbose=True, dpi=300):
    """Saves the matplotlib figure."""
    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        if verbose:
            print(f"Saved figure: {output_path}")
        plt.close(fig) # Close figure to free memory
    except Exception as e:
        print(f"Warning: Failed to save figure {output_path}. Error: {e}")

def load_pickle(filepath, verbose=True):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    if verbose:
        print(f"Loading data from: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        raise

def calculate_path_metrics(coords):
    """Calculates path length and displacement for a sequence of coordinates."""
    if coords is None or len(coords) < 2:
        return 0.0, 0.0 # Cannot calculate for single point or empty

    # Path length: sum of distances between consecutive points
    diffs = np.diff(coords, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(step_distances)

    # Displacement: distance between start and end points
    displacement_vec = coords[-1] - coords[0]
    displacement = np.linalg.norm(displacement_vec)

    return path_length, displacement

# --- Trajectory Loading and Feature Extraction ---
# Reusing functions from your provided script
def load_trajectory(traj_path, top_path=None, verbose=True):
    """
    Loads trajectory. If traj_path is a PDB, it's used for both topology and coordinates.
    Otherwise, top_path is required. Uses in_memory=True.
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
            # in_memory=True is crucial for efficient feature extraction later
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
            # in_memory=True is crucial
            u = mda.Universe(top_path, traj_path, in_memory=True)
            num_atoms = len(u.atoms)

            # Check atom count consistency
            if len(u.trajectory) > 0 and u.atoms.n_atoms != num_top_atoms:
                 raise ValueError(f"Atom count mismatch in {traj_path}: Topology has {num_top_atoms}, "
                                  f"but trajectory frame reports {u.atoms.n_atoms}.")

        if len(u.trajectory) == 0:
            raise ValueError(f"Trajectory {traj_path} loaded with 0 frames.")

        if verbose:
            print(f"Successfully loaded Trajectory ({len(u.atoms)} atoms, {len(u.trajectory)} frames)")
        return u

    except IndexError as ie:
        print(f"\nError: IndexError encountered while reading trajectory '{os.path.basename(traj_path)}'. Check consistency between topology and trajectory files.")
        print(f"Original error: {ie}")
        raise # Re-raise the exception
    except MemoryError:
        print(f"MemoryError loading {traj_path}. Trajectory too large for memory.")
        raise
    except Exception as e:
        print(f"Error loading trajectory {traj_path}: {e}")
        raise

def extract_trajectory_features(universe, pairs_indices, selection="name CA", verbose=True):
    """Extracts distances for selected Cα pairs over a trajectory."""
    if verbose: print(f"Extracting features using selection '{selection}'...")
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_atoms_selected = len(atoms)
    n_pairs = len(pairs_indices)

    if n_atoms_selected == 0:
        raise ValueError(f"Atom selection '{selection}' yielded 0 atoms in the trajectory.")
    if n_pairs == 0:
        raise ValueError("Cannot extract features: No pairs indices provided.")

    features = np.empty((n_frames, n_pairs), dtype=np.float32)
    pair_indices_array = np.array(pairs_indices, dtype=int)

    # Check if any pair index is out of bounds for the selected atoms *BEFORE* loop
    max_index_required = pair_indices_array.max()
    if max_index_required >= n_atoms_selected:
         raise IndexError(f"Maximum selected pair index ({max_index_required}) is out of bounds "
                          f"for the selected atoms group (size {n_atoms_selected}) with selection '{selection}'. "
                          "Ensure selection and pairs indices are consistent with the trajectory.")

    # Iterate through the trajectory (already in memory)
    # Use atoms.positions which accesses coordinates for the current frame
    for i, ts in enumerate(universe.trajectory):
        coords = atoms.positions # Gets coordinates for the current frame 'ts'
        # Use the faster distance_array for pairs
        # Note: This calculates more distances than needed, but is often faster than looping python-side
        # We index directly into the full distance matrix calculated for the frame
        try:
             # Get distances for specific pairs: atom_i vs atom_j for all (i,j) in pairs
             distances = distance_array(coords[pair_indices_array[:, 0]],
                                        coords[pair_indices_array[:, 1]],
                                        box=ts.dimensions).diagonal() # diagonal gives dist(i,i) if same points passed, need separate points
             # Correction: Need to calculate dist(coords[idx1], coords[idx2]) for each pair (idx1, idx2)
             # Let's loop for clarity, or use advanced indexing if performance critical
             frame_distances = []
             for idx1, idx2 in pair_indices_array:
                  dist = np.linalg.norm(coords[idx1] - coords[idx2])
                  frame_distances.append(dist)
             features[i, :] = frame_distances

             # Alternative using distance_array (might be faster for many pairs)
             # full_dist_matrix_frame = distance_array(coords, coords, box=ts.dimensions)
             # features[i, :] = full_dist_matrix_frame[pair_indices_array[:, 0], pair_indices_array[:, 1]]

        except IndexError as e:
             print(f"\nIndexError during feature extraction at frame {i}. This should not happen after the initial check.")
             print(f"Max required index: {max_index_required}, Atoms selected: {n_atoms_selected}")
             print(f"Pair indices causing error might be: {pair_indices_array[pair_indices_array >= n_atoms_selected]}")
             raise e

    if verbose: print(f"Finished extracting features. Shape: {features.shape}")
    return features

def load_static_structure(pdb_path, selection="name CA", verbose=True):
    """Loads a static PDB structure (similar to load_trajectory but for single frame)."""
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if verbose:
        print(f"Loading static structure: {pdb_path}")
    try:
        u = mda.Universe(pdb_path)
        atoms = u.select_atoms(selection)
        if len(atoms) == 0:
            raise ValueError(f"Selection '{selection}' yielded 0 atoms in {pdb_path}")
        if verbose:
            print(f"  Selected {len(atoms)} atoms using '{selection}'")
        return u, atoms
    except Exception as e:
        print(f"Error loading static structure {pdb_path}: {e}")
        raise

def extract_static_features(atoms, pairs_indices):
    """Extracts distances for selected Cα pairs for a static structure."""
    n_atoms_selected = len(atoms)
    n_pairs = len(pairs_indices)
    if n_pairs == 0:
        raise ValueError("Cannot extract features: No pairs indices provided.")

    features = np.empty((1, n_pairs), dtype=np.float32) # Shape (1, n_pairs)
    pair_indices_array = np.array(pairs_indices, dtype=int)

    max_index_required = pair_indices_array.max()
    if max_index_required >= n_atoms_selected:
         raise IndexError(f"Maximum selected pair index ({max_index_required}) is out of bounds "
                          f"for the static atoms group (size {n_atoms_selected}).")

    coords = atoms.positions
    # Loop for clarity or use advanced indexing
    static_distances = []
    for idx1, idx2 in pair_indices_array:
         dist = np.linalg.norm(coords[idx1] - coords[idx2])
         static_distances.append(dist)
    features[0, :] = static_distances

    # Alternative using distance_array
    # full_dist_matrix_static = distance_array(coords, coords)
    # features[0, :] = full_dist_matrix_static[pair_indices_array[:, 0], pair_indices_array[:, 1]]

    return features

# --- PCA Plotting ---
def plot_projected_paths(proj1, proj2, proj_start, proj_end, output_path, title_suffix="", verbose=True):
    """Plots the projected trajectories in the first 2 PCA dimensions with fixed labels."""
    fig, ax = plt.subplots(figsize=(9, 7)) # Adjusted figsize slightly
    # Plot trajectories with reduced marker size and line width
    ax.plot(proj1[:, 0], proj1[:, 1], '-', label='Traj 1 (SMD)', color='C0', alpha=0.4, linewidth=1.5, zorder=2)
    ax.plot(proj2[:, 0], proj2[:, 1], '-', label='Traj 2 (ProToken)', color='C1', alpha=0.5, linewidth=1.5, zorder=3)
    # Mark trajectory ends clearly
    ax.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='SMD Start', zorder=4)
    ax.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='SMD End', zorder=4)
    ax.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='ProToken Start', zorder=5)
    ax.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='ProToken End', zorder=5)
    # Mark reference points
    ax.plot(proj_start[:, 0], proj_start[:, 1], 'P', color='green', markersize=12, label='Ref Start PDB', zorder=10, mec='black')
    ax.plot(proj_end[:, 0], proj_end[:, 1], 'X', color='red', markersize=12, label='Ref End PDB', zorder=10, mec='black')

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    title = f"Projected Trajectories (PC1 vs PC2){title_suffix}"
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    ax.grid(True, linestyle='--', alpha=0.6)

    # Place legend outside the figure
    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Leave space on the right

    save_figure(fig, output_path, verbose)

# --- DTW Calculation and Plotting ---
def calculate_and_plot_dtw(proj1, proj2, proj_start, proj_end, output_path, verbose=True):
    """Calculates DTW and plots the alignment visualization with fixed trajectory labels."""
    if verbose:
        print("\nRecalculating DTW distance and path on PCA components...")
    # Ensure data is C-contiguous double for dtaidistance
    proj1_dtw = np.ascontiguousarray(proj1, dtype=np.double)
    proj2_dtw = np.ascontiguousarray(proj2, dtype=np.double)

    # Use a window size that is a fraction of the longer trajectory, e.g., 10%
    # Or a fixed number, or based on time difference if available.
    # A window is important for performance with long trajectories like SMD.
    window_arg = max(10, int(0.1 * max(len(proj1_dtw), len(proj2_dtw))))
    if verbose: print(f"  Using DTW window size: {window_arg}")

    try:
        # Use dtw_ndim for multi-dimensional DTW
        distance, paths = dtw_ndim.warping_paths(proj1_dtw, proj2_dtw, window=window_arg, psi=0) # psi=0 means no relaxation at ends
        best_path = dtw.best_path(paths) # Extract the best path
        if verbose:
            print(f"DTW calculation complete. Distance = {distance:.4f}")
    except Exception as e:
         print(f"Warning: DTW calculation failed: {e}.")
         return np.inf, None # Return infinity and None path on failure


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 7)) # Adjusted figsize

    # Plot trajectories faintly
    ax.plot(proj1[:, 0], proj1[:, 1], '-', color='C0', alpha=0.3, linewidth=1, label='Traj 1 (SMD)')
    ax.plot(proj2[:, 0], proj2[:, 1], '-', color='C1', alpha=0.3, linewidth=1, label='Traj 2 (ProToken)')

    # Plot DTW alignment paths (lines connecting aligned points)
    if best_path:
        for idx1, idx2 in best_path[::max(1, len(best_path)//200)]: # Plot subset of lines for clarity if too many
            if idx1 < len(proj1) and idx2 < len(proj2):
                 ax.plot([proj1[idx1, 0], proj2[idx2, 0]],
                         [proj1[idx1, 1], proj2[idx2, 1]],
                         '-', color='gray', linewidth=0.5, alpha=0.4)
    else:
         print("Warning: No DTW path available for plotting.")

    # Mark trajectory start and end points clearly
    ax.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='SMD Start')
    ax.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='SMD End')
    ax.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='ProToken Start')
    ax.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='ProToken End')

    # Add reference structure markers
    ax.plot(proj_start[:, 0], proj_start[:, 1], 'P', color='green', markersize=12, label='Ref Start PDB', zorder=10, mec='black')
    ax.plot(proj_end[:, 0], proj_end[:, 1], 'X', color='red', markersize=12, label='Ref End PDB', zorder=10, mec='black')

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"DTW Alignment in PCA Space (Distance={distance:.2f})")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Place legend outside
    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    save_figure(fig, output_path, verbose)
    return distance, best_path # Return the calculated best path

# --- Metastable State Identification Logic ---
def find_metastable_candidates(proj1, proj2, best_path, args):
    """
    Identifies potential metastable states based on DTW alignment and SMD path analysis.
    (Reusing logic from previous response, adapted for current inputs)
    """
    n_frames_smd = proj1.shape[0]
    n_frames_pro = proj2.shape[0]

    if args.verbose:
        print("\n--- Identifying Potential Metastable States ---")
        print(f"Using DTW path to analyze SMD ({n_frames_smd} frames) vs ProToken ({n_frames_pro} frames)")

    # 1. Build ProToken to SMD mapping
    pro_to_smd_map = defaultdict(list)
    if not best_path:
         print("Warning: DTW path is empty or None. Cannot identify metastable states.")
         return [], np.zeros(n_frames_pro)

    for idx_smd, idx_pro in best_path:
        if 0 <= idx_smd < n_frames_smd and 0 <= idx_pro < n_frames_pro:
             pro_to_smd_map[idx_pro].append(idx_smd)
        # else: # Optional warning for out-of-bounds indices from DTW
             # print(f"Warning: DTW path index out of bounds...")

    # 2. Calculate Path/Displacement Ratio for non-terminal ProToken frames
    ratios = np.zeros(n_frames_pro)
    end_frame_threshold_pro = int(n_frames_pro * (1 - args.exclude_end_fraction))
    if args.verbose:
        print(f"Analyzing ProToken frames 0 to {end_frame_threshold_pro - 1} (excluding last {args.exclude_end_fraction*100:.0f}%).")
        print(f"Criteria: min_smd_frames={args.min_smd_frames}, min_ratio={args.min_ratio}, peak_prominence={args.peak_prominence}")

    for j in range(end_frame_threshold_pro):
        if j not in pro_to_smd_map: continue
        smd_indices = sorted(list(set(pro_to_smd_map[j])))
        if len(smd_indices) < args.min_smd_frames: continue

        smd_segment_coords = proj1[smd_indices, :args.n_components_dtw] # Use first N components for calc
        path_length, displacement = calculate_path_metrics(smd_segment_coords)
        ratio = path_length / (displacement + EPSILON)
        ratios[j] = ratio
        if args.verbose > 1:
             print(f"  ProTkn {j}: SMD Seg=[{min(smd_indices)}-{max(smd_indices)}] (N={len(smd_indices)}), L={path_length:.2f}, D={displacement:.2f}, R={ratio:.2f}")


    # 3. Find peaks in the ratio profile
    peaks, properties = find_peaks(ratios, height=args.min_ratio, prominence=args.peak_prominence)
    if args.verbose:
        print(f"Found {len(peaks)} peaks in ratio profile meeting criteria.")

    # 4. Extract Representative Frames for each peak
    metastable_candidates = []
    for j_peak in peaks:
        pro_rep_idx = j_peak
        smd_indices_peak = sorted(list(set(pro_to_smd_map[j_peak])))
        if not smd_indices_peak: continue # Should have indices if it's a peak
        # Use median SMD index as representative
        smd_rep_idx = smd_indices_peak[len(smd_indices_peak) // 2]
        metastable_candidates.append((smd_rep_idx, pro_rep_idx, ratios[j_peak]))
        if args.verbose:
             print(f"  Candidate: ProTkn Frame={pro_rep_idx} (Ratio={ratios[j_peak]:.2f}) <-> Median SMD Frame={smd_rep_idx} (Range: {min(smd_indices_peak)}-{max(smd_indices_peak)})")

    # 5. Optional: Plot the ratio profile
    if args.plot_ratios:
         fig, ax = plt.subplots(figsize=(10, 5))
         ax.plot(range(n_frames_pro), ratios, '.-', label='Path/Displacement Ratio', color='purple', markersize=4)
         ax.plot(peaks, ratios[peaks], "x", color='red', markersize=10, mew=2, label='Identified Peaks') # Make peaks more visible
         ax.axhline(args.min_ratio, color='gray', linestyle='--', label=f'Min Ratio ({args.min_ratio})')
         ax.set_xlabel("ProToken Frame Index")
         ax.set_ylabel("SMD Path Length / Displacement Ratio")
         ax.set_title("SMD Trajectory Wandering Metric per ProToken Frame")
         ax.legend()
         ax.grid(True, linestyle='--', alpha=0.6)
         ax.set_xlim(0, n_frames_pro -1) # Ensure x-axis limits are sensible
         ratio_plot_path = os.path.join(args.output_dir, "smd_wandering_ratio_profile.png")
         save_figure(fig, ratio_plot_path, args.verbose)

    return metastable_candidates, ratios

# --- Frame Extraction ---
def extract_and_save_frame(universe, frame_index, output_filename, selection="protein", verbose=True):
    """Extracts a specific frame and saves it as a PDB."""
    if frame_index < 0 or frame_index >= len(universe.trajectory):
        print(f"Warning: Frame index {frame_index} is out of bounds for trajectory (Length: {len(universe.trajectory)}). Skipping extraction for {output_filename}.")
        return False
    try:
        # Go to the desired frame
        universe.trajectory[frame_index]
        # Select atoms for output (e.g., "protein", "all")
        atoms_to_write = universe.select_atoms(selection)
        if len(atoms_to_write) == 0:
             print(f"Warning: Selection '{selection}' yielded 0 atoms for frame {frame_index}. Cannot write PDB {output_filename}.")
             return False
        # Write the selected atoms of the current frame
        with mda.Writer(output_filename, atoms_to_write.n_atoms) as W:
            W.write(atoms_to_write)
        if verbose:
            print(f"Saved frame {frame_index} (selection: '{selection}') to: {output_filename}")
        return True
    except Exception as e:
        print(f"Error extracting/saving frame {frame_index} to {output_filename}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify metastable states from SMD/ProToken comparison using pre-computed features/PCA, and extract representative frames."
    )
    # Input Files - REQUIRED
    parser.add_argument("traj1", help="Path to the first trajectory file (SMD PDB or other format).")
    parser.add_argument("traj2", help="Path to the second trajectory file (ProToken PDB or other format).")
    parser.add_argument("start_pdb", help="Path to the reference starting structure PDB file.")
    parser.add_argument("end_pdb", help="Path to the reference ending structure PDB file.")
    parser.add_argument("selected_pairs_indices_pkl", help="Path to the .pkl file containing selected Cα pair indices.")
    parser.add_argument("pca_model_pkl", help="Path to the saved .pkl file of the fitted PCA model.")
    # Input Files - OPTIONAL
    parser.add_argument("--topology1", default=None, help="Optional topology for traj1 (required if traj1 is not PDB).")
    parser.add_argument("--topology2", default=None, help="Optional topology for traj2 (required if traj2 is not PDB).")
    # Parameters - Atom Selection & PCA
    parser.add_argument("-s", "--atom_selection", default="protein and name CA", help="Atom selection for Cα atoms used for distances/PCA.")
    parser.add_argument("--n_components_dtw", type=int, default=3, help="Number of PCA components to use for DTW calculation.")
    # Parameters - Metastable Identification
    parser.add_argument("--min_smd_frames", type=int, default=10, help="Min SMD frames mapped to a ProToken frame.")
    parser.add_argument("--exclude_end_fraction", type=float, default=0.15, help="Fraction of ProToken end to exclude (0.0 to <1.0).")
    parser.add_argument("--min_ratio", type=float, default=3.0, help="Min Path/Displacement ratio for peaks.")
    parser.add_argument("--peak_prominence", type=float, default=1.0, help="Required peak prominence in ratio profile.")
    # Parameters - Output & Frame Extraction
    parser.add_argument("-o", "--output_dir", default="metastable_analysis_frames", help="Directory to save outputs.")
    parser.add_argument("--no_plot_pca", action="store_false", dest="plot_pca", help="Skip generating the PCA projection plot.")
    parser.add_argument("--no_plot_dtw", action="store_false", dest="plot_dtw", help="Skip generating the DTW alignment plot.")
    parser.add_argument("--no_plot_ratios", action="store_false", dest="plot_ratios", help="Skip generating the wandering ratio profile plot.")
    parser.add_argument("--extract_selection", default="protein", help="Atom selection for writing extracted PDB frames (e.g., 'protein', 'all').")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity level (-v, -vv).")

    args = parser.parse_args()

    # --- Validate Args ---
    if not 0.0 <= args.exclude_end_fraction < 1.0:
        parser.error("--exclude_end_fraction must be between 0.0 and 1.0 (exclusive of 1.0).")
    if args.n_components_dtw < 1:
        parser.error("--n_components_dtw must be at least 1.")

    try:
        # --- 0. Setup ---
        make_output_dir(args.output_dir)
        if args.verbose:
            print(f"Starting analysis. Outputs will be saved to: {args.output_dir}")

        # --- 1. Load Pre-computed Data & Models ---
        selected_pairs_indices = load_pickle(args.selected_pairs_indices_pkl, args.verbose)
        pca_model = load_pickle(args.pca_model_pkl, args.verbose)
        # Verify loaded types
        if not isinstance(selected_pairs_indices, list): raise TypeError("Loaded pairs indices is not a list.")
        if not hasattr(pca_model, 'transform'): raise TypeError("Loaded PCA model does not have a 'transform' method.")
        # Store the number of components the loaded PCA model actually has
        n_components_model = pca_model.n_components_
        if args.verbose: print(f"Loaded PCA model with {n_components_model} components.")
        if args.n_components_dtw > n_components_model:
             print(f"Warning: Requested {args.n_components_dtw} components for DTW, but PCA model only has {n_components_model}. Using {n_components_model}.")
             args.n_components_dtw = n_components_model


        # --- 2. Load Trajectories and Static Structures ---
        u1 = load_trajectory(args.traj1, args.topology1, args.verbose) # SMD
        u2 = load_trajectory(args.traj2, args.topology2, args.verbose) # ProToken
        ref_start_u, ref_start_atoms = load_static_structure(args.start_pdb, args.atom_selection, args.verbose)
        ref_end_u, ref_end_atoms = load_static_structure(args.end_pdb, args.atom_selection, args.verbose)

        # --- 3. Extract Features ---
        if args.verbose: print("\nExtracting features for trajectories using loaded pairs...")
        features1 = extract_trajectory_features(u1, selected_pairs_indices, args.atom_selection, args.verbose > 0)
        features2 = extract_trajectory_features(u2, selected_pairs_indices, args.atom_selection, args.verbose > 0)
        if args.verbose: print("Extracting features for static start/end points...")
        features_start = extract_static_features(ref_start_atoms, selected_pairs_indices)
        features_end = extract_static_features(ref_end_atoms, selected_pairs_indices)

        # --- 4. Project Features using Loaded PCA Model ---
        if args.verbose: print("\nProjecting features into PCA space using loaded model...")
        proj1 = pca_model.transform(features1)
        proj2 = pca_model.transform(features2)
        proj_start = pca_model.transform(features_start)
        proj_end = pca_model.transform(features_end)
        if args.verbose:
             print(f"Projected shapes: Traj1={proj1.shape}, Traj2={proj2.shape}, Start={proj_start.shape}, End={proj_end.shape}")

        # --- 5. Optional: Plot PCA Projection ---
        if args.plot_pca:
            plot_projected_paths(
                proj1, proj2, proj_start, proj_end,
                os.path.join(args.output_dir, "pca_projection_check.png"),
                title_suffix=f" ({len(selected_pairs_indices)} features)",
                verbose=args.verbose
            )

        # --- 6. Recalculate DTW ---
        dtw_distance, best_path = calculate_and_plot_dtw(
            proj1[:, :args.n_components_dtw], # Use specified components for DTW
            proj2[:, :args.n_components_dtw],
            proj_start[:, :args.n_components_dtw], # Also project ref points for plotting context
            proj_end[:, :args.n_components_dtw],
            os.path.join(args.output_dir, "dtw_alignment_recalculated.png") if args.plot_dtw else None, # Only plot if requested
            verbose=args.verbose
        )
        if not best_path:
             print("\nDTW calculation failed or yielded no path. Cannot proceed with metastable analysis.")
             exit() # Exit if DTW failed
        print(f"\nRecalculated DTW Distance: {dtw_distance:.4f}")

        # --- 7. Identify Metastable Candidates ---
        candidates, _ = find_metastable_candidates(
            proj1, proj2, best_path, args # Pass full projections for context, use n_components_dtw inside function
        )

        # --- 8. Report and Extract Frames ---
        print("\n--- Final Identified Potential Metastable State Pairs ---")
        if candidates:
            print("(SMD_Representative_Frame, ProToken_Frame, PathDispRatio_at_Peak)")
            candidate_info = []
            for smd_idx, pro_idx, ratio_val in candidates:
                print(f"({smd_idx}, {pro_idx}, {ratio_val:.2f})")
                candidate_info.append(f"{smd_idx}\t{pro_idx}\t{ratio_val:.2f}\n")

                # Extract Frames
                smd_out_pdb = os.path.join(args.output_dir, f"metastable_smd_frame_{smd_idx}.pdb")
                pro_out_pdb = os.path.join(args.output_dir, f"metastable_protoken_frame_{pro_idx}.pdb")

                if args.verbose: print(f"  Extracting SMD frame {smd_idx}...")
                extract_and_save_frame(u1, smd_idx, smd_out_pdb, args.extract_selection, args.verbose > 1)

                if args.verbose: print(f"  Extracting ProToken frame {pro_idx}...")
                extract_and_save_frame(u2, pro_idx, pro_out_pdb, args.extract_selection, args.verbose > 1)

            # Save candidates list to file
            candidates_file = os.path.join(args.output_dir, "metastable_candidate_frames.txt")
            with open(candidates_file, 'w') as f:
                 f.write("# Potential Metastable State Representative Frames\n")
                 f.write("# SMD_Representative_Frame\tProToken_Frame\tPathDispRatio_at_Peak\n")
                 f.writelines(candidate_info)
            print(f"\nSaved candidate frame pairs to: {candidates_file}")
            print(f"Extracted PDB frames saved in: {args.output_dir}")

        else:
            print("No potential metastable states identified with the current criteria.")


    except FileNotFoundError as fnf_err:
         print(f"\nError: Input file not found. {fnf_err}")
    except (TypeError, ValueError, IndexError) as data_err:
        print(f"\nError during data processing or consistency check: {data_err}")
        import traceback
        traceback.print_exc()
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()

    print("\nAnalysis complete.")
