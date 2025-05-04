#!/usr/bin/env python3

import MDAnalysis as mda
import numpy as np
from sklearn.decomposition import PCA # For type hints
from dtaidistance import dtw_ndim, dtw
import warnings
import argparse
import os
import pickle
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from MDAnalysis.lib.distances import distance_array

# --- Plotting Setup ---
import matplotlib
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Using Agg backend for Matplotlib.")
except ImportError:
    import matplotlib.pyplot as plt
    print("Agg backend not available, using default Matplotlib backend.")

# --- Constants ---
EPSILON = 1e-6

# --- Helper Functions (reuse previous) ---
def make_output_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def save_figure(fig, output_path, verbose=True, dpi=300):
    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        if verbose:
            print(f"Saved figure: {output_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Failed to save figure {output_path}. Error: {e}")

def load_pickle(filepath, verbose=True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    if verbose: print(f"Loading data from: {filepath}")
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        return data
    except Exception as e: print(f"Error loading pickle file {filepath}: {e}"); raise

def calculate_path_metrics(coords):
    if coords is None or len(coords) < 2: return 0.0, 0.0
    diffs = np.diff(coords, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(step_distances)
    displacement_vec = coords[-1] - coords[0]
    displacement = np.linalg.norm(displacement_vec)
    return path_length, displacement

# --- Trajectory Loading and Feature Extraction (reuse previous) ---
def load_trajectory(traj_path, top_path=None, verbose=True):
    if not os.path.exists(traj_path): raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    is_pdb = traj_path.lower().endswith(('.pdb', '.ent'))
    try:
        if is_pdb:
            if top_path and verbose: print(f"  Info: PDB Trajectory '{os.path.basename(traj_path)}'. Ignoring topology '{os.path.basename(top_path)}'.")
            if verbose: print(f"Loading PDB Trajectory (topology & coords): '{traj_path}'")
            print("  Loading PDB structure and coordinates into memory...")
            u = mda.Universe(traj_path, in_memory=True)
        else:
            if not top_path: raise ValueError(f"Trajectory '{traj_path}' needs topology (--topology1/--topology2).")
            if not os.path.exists(top_path): raise FileNotFoundError(f"Topology not found for {traj_path}: {top_path}")
            if verbose: print(f"Loading Trajectory: Topology='{top_path}', Trajectory='{traj_path}'")
            print("  Loading topology and trajectory coordinates into memory...")
            u = mda.Universe(top_path, traj_path, in_memory=True)
            # Consistency check
            if len(u.trajectory)>0:
                structure_u = mda.Universe(top_path) # Reload topology for check
                if u.atoms.n_atoms != structure_u.atoms.n_atoms:
                     raise ValueError(f"Atom count mismatch: Topology ({structure_u.atoms.n_atoms}) vs Trajectory ({u.atoms.n_atoms}).")
        if len(u.trajectory) == 0: raise ValueError(f"Trajectory {traj_path} loaded with 0 frames.")
        if verbose: print(f"Successfully loaded Trajectory ({len(u.atoms)} atoms, {len(u.trajectory)} frames)")
        return u
    except Exception as e: print(f"Error loading trajectory {traj_path}: {e}"); raise

def extract_trajectory_features(universe, pairs_indices, selection="name CA", verbose=True):
    if verbose: print(f"Extracting features using selection '{selection}'...")
    atoms = universe.select_atoms(selection)
    n_frames, n_atoms_sel, n_pairs = len(universe.trajectory), len(atoms), len(pairs_indices)
    if n_atoms_sel == 0: raise ValueError(f"Selection '{selection}' yielded 0 atoms.")
    if n_pairs == 0: raise ValueError("No pairs indices provided.")
    features = np.empty((n_frames, n_pairs), dtype=np.float32)
    pair_indices_array = np.array(pairs_indices, dtype=int)
    max_idx = pair_indices_array.max()
    if max_idx >= n_atoms_sel: raise IndexError(f"Max pair index ({max_idx}) out of bounds for selection ({n_atoms_sel}).")
    for i, ts in enumerate(universe.trajectory):
        coords = atoms.positions
        frame_dists = [np.linalg.norm(coords[idx1] - coords[idx2]) for idx1, idx2 in pair_indices_array]
        features[i, :] = frame_dists
    if verbose: print(f"Finished extracting features. Shape: {features.shape}")
    return features

def load_static_structure(pdb_path, selection="name CA", verbose=True):
    if not os.path.exists(pdb_path): raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if verbose: print(f"Loading static structure: {pdb_path}")
    try:
        u = mda.Universe(pdb_path); atoms = u.select_atoms(selection)
        if len(atoms) == 0: raise ValueError(f"Selection '{selection}' yielded 0 atoms in {pdb_path}")
        if verbose: print(f"  Selected {len(atoms)} atoms.")
        return u, atoms
    except Exception as e: print(f"Error loading static structure {pdb_path}: {e}"); raise

def extract_static_features(atoms, pairs_indices):
    n_atoms_sel, n_pairs = len(atoms), len(pairs_indices)
    if n_pairs == 0: raise ValueError("No pairs indices provided.")
    features = np.empty((1, n_pairs), dtype=np.float32)
    pair_indices_array = np.array(pairs_indices, dtype=int)
    max_idx = pair_indices_array.max()
    if max_idx >= n_atoms_sel: raise IndexError(f"Max pair index ({max_idx}) out of bounds for static atoms ({n_atoms_sel}).")
    coords = atoms.positions
    static_dists = [np.linalg.norm(coords[idx1] - coords[idx2]) for idx1, idx2 in pair_indices_array]
    features[0, :] = static_dists
    return features

# --- Plotting (reuse previous) ---
def plot_projected_paths(proj1, proj2, proj_start, proj_end, output_path, title_suffix="", verbose=True):
    if not output_path: return # Skip plotting if no path provided
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(proj1[:, 0], proj1[:, 1], '-', label='Traj 1 (SMD)', color='C0', alpha=0.4, linewidth=1.5, zorder=2)
    ax.plot(proj2[:, 0], proj2[:, 1], '-', label='Traj 2 (ProToken)', color='C1', alpha=0.5, linewidth=1.5, zorder=3)
    ax.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='SMD Start', zorder=4)
    ax.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='SMD End', zorder=4)
    ax.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='ProToken Start', zorder=5)
    ax.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='ProToken End', zorder=5)
    ax.plot(proj_start[:, 0], proj_start[:, 1], 'P', color='green', markersize=12, label='Ref Start PDB', zorder=10, mec='black')
    ax.plot(proj_end[:, 0], proj_end[:, 1], 'X', color='red', markersize=12, label='Ref End PDB', zorder=10, mec='black')
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    title = f"Projected Trajectories (PC1 vs PC2){title_suffix}"
    ax.set_title(title); ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_figure(fig, output_path, verbose)

def calculate_and_plot_dtw(proj1, proj2, proj_start, proj_end, output_path, verbose=True):
    if verbose: print("\nRecalculating DTW distance and path on PCA components...")
    proj1_dtw = np.ascontiguousarray(proj1, dtype=np.double)
    proj2_dtw = np.ascontiguousarray(proj2, dtype=np.double)
    window_arg = max(10, int(0.1 * max(len(proj1_dtw), len(proj2_dtw))))
    if verbose: print(f"  Using DTW window size: {window_arg}")
    try:
        distance, paths = dtw_ndim.warping_paths(proj1_dtw, proj2_dtw, window=window_arg, psi=0)
        best_path = dtw.best_path(paths)
        if verbose: print(f"DTW calculation complete. Distance = {distance:.4f}")
    except Exception as e: print(f"Warning: DTW calculation failed: {e}."); return np.inf, None

    if output_path: # Only plot if path provided
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(proj1[:, 0], proj1[:, 1], '-', color='C0', alpha=0.3, linewidth=1, label='Traj 1 (SMD)')
        ax.plot(proj2[:, 0], proj2[:, 1], '-', color='C1', alpha=0.3, linewidth=1, label='Traj 2 (ProToken)')
        if best_path:
            plot_step = max(1, len(best_path)//200) # Optimize plotting
            for idx1, idx2 in best_path[::plot_step]:
                if idx1 < len(proj1) and idx2 < len(proj2):
                     ax.plot([proj1[idx1, 0], proj2[idx2, 0]], [proj1[idx1, 1], proj2[idx2, 1]], '-', color='gray', linewidth=0.5, alpha=0.4)
        ax.plot(proj1[0, 0], proj1[0, 1], 'o', color='blue', markersize=8, label='SMD Start')
        ax.plot(proj1[-1, 0], proj1[-1, 1], '^', color='blue', markersize=8, label='SMD End')
        ax.plot(proj2[0, 0], proj2[0, 1], 'o', color='orange', markersize=8, label='ProToken Start')
        ax.plot(proj2[-1, 0], proj2[-1, 1], '^', color='orange', markersize=8, label='ProToken End')
        ax.plot(proj_start[:, 0], proj_start[:, 1], 'P', color='green', markersize=12, label='Ref Start PDB', zorder=10, mec='black')
        ax.plot(proj_end[:, 0], proj_end[:, 1], 'X', color='red', markersize=12, label='Ref End PDB', zorder=10, mec='black')
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title(f"DTW Alignment in PCA Space (Distance={distance:.2f})")
        ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5), framealpha=0.9)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_figure(fig, output_path, verbose)
    return distance, best_path

# --- Metastable State Identification (reuse previous) ---
def find_metastable_candidates(proj1, proj2, best_path, args):
    n_frames_smd, n_frames_pro = proj1.shape[0], proj2.shape[0]
    if args.verbose: print("\n--- Identifying Potential Metastable States ---")
    pro_to_smd_map = defaultdict(list)
    if not best_path: print("Warning: DTW path empty."); return [], np.zeros(n_frames_pro)
    for idx_smd, idx_pro in best_path:
        if 0 <= idx_smd < n_frames_smd and 0 <= idx_pro < n_frames_pro: pro_to_smd_map[idx_pro].append(idx_smd)
    ratios = np.zeros(n_frames_pro)
    end_frame_thresh = int(n_frames_pro * (1 - args.exclude_end_fraction))
    if args.verbose: print(f"Analyzing ProToken frames 0 to {end_frame_thresh - 1}. Criteria: min_smd={args.min_smd_frames}, min_R={args.min_ratio}, prominence={args.peak_prominence}")
    for j in range(end_frame_thresh):
        if j not in pro_to_smd_map: continue
        smd_indices = sorted(list(set(pro_to_smd_map[j])))
        if len(smd_indices) < args.min_smd_frames: continue
        smd_seg_coords = proj1[smd_indices, :args.n_components_dtw]
        path_len, disp = calculate_path_metrics(smd_seg_coords)
        ratio = path_len / (disp + EPSILON)
        ratios[j] = ratio
        if args.verbose > 1: print(f"  ProTkn {j}: SMD Seg=[{min(smd_indices)}-{max(smd_indices)}] (N={len(smd_indices)}), R={ratio:.2f}")
    peaks, _ = find_peaks(ratios, height=args.min_ratio, prominence=args.peak_prominence)
    if args.verbose: print(f"Found {len(peaks)} peaks meeting criteria.")
    metastable_candidates = []
    for j_peak in peaks:
        pro_rep_idx = j_peak
        smd_indices_peak = sorted(list(set(pro_to_smd_map[j_peak])))
        if not smd_indices_peak: continue
        # Keep all SMD indices for this peak
        metastable_candidates.append((smd_indices_peak, pro_rep_idx, ratios[j_peak]))
        if args.verbose: print(f"  Candidate: ProTkn Frame={pro_rep_idx} (Ratio={ratios[j_peak]:.2f}) <-> SMD Frames ({len(smd_indices_peak)}): {min(smd_indices_peak)}-{max(smd_indices_peak)}")

    # --- Plotting Ratios (Modified Default Behavior) ---
    if not args.no_plot_ratios: # Plot unless specifically told not to
         fig, ax = plt.subplots(figsize=(10, 5))
         ax.plot(range(n_frames_pro), ratios, '.-', label='Path/Disp Ratio', color='purple', markersize=4)
         ax.plot(peaks, ratios[peaks], "x", color='red', markersize=10, mew=2, label='Peaks')
         ax.axhline(args.min_ratio, color='gray', linestyle='--', label=f'Min Ratio ({args.min_ratio})')
         ax.set_xlabel("ProToken Frame Index"); ax.set_ylabel("SMD Path Length / Displacement Ratio")
         ax.set_title("SMD Trajectory Wandering Metric per ProToken Frame")
         ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim(0, n_frames_pro -1)
         ratio_plot_path = os.path.join(args.output_dir, "smd_wandering_ratio_profile.png")
         save_figure(fig, ratio_plot_path, args.verbose > 0) # Use verbose flag for print

    return metastable_candidates, ratios

# --- Frame Extraction (Modified for Clusters) ---
def extract_and_save_frame(universe, frame_index, output_filename, selection="protein", verbose=True):
    """Extracts a single frame and saves it as a PDB."""
    # (Same as previous version)
    if frame_index < 0 or frame_index >= len(universe.trajectory):
        print(f"Warning: Frame index {frame_index} out of bounds ({len(universe.trajectory)} frames). Skipping {output_filename}.")
        return False
    try:
        universe.trajectory[frame_index]
        atoms_to_write = universe.select_atoms(selection)
        if len(atoms_to_write) == 0:
             print(f"Warning: Selection '{selection}' yielded 0 atoms for frame {frame_index}. Cannot write {output_filename}.")
             return False
        with mda.Writer(output_filename, atoms_to_write.n_atoms) as W: W.write(atoms_to_write)
        if verbose: print(f"  Saved frame {frame_index} (sel: '{selection}') to: {os.path.basename(output_filename)}")
        return True
    except Exception as e: print(f"Error extracting/saving frame {frame_index} to {output_filename}: {e}"); return False

def extract_and_save_frame_cluster(universe, frame_indices, output_filename, selection="protein", verbose=True):
    """Extracts multiple frames (cluster) and saves as a multi-MODEL PDB."""
    if not frame_indices:
        print(f"Warning: No frame indices provided for cluster. Skipping {output_filename}.")
        return False
    try:
        # Select atoms ONCE
        atoms_to_write = universe.select_atoms(selection)
        if len(atoms_to_write) == 0:
             print(f"Warning: Selection '{selection}' yielded 0 atoms. Cannot write cluster {output_filename}.")
             return False

        with mda.Writer(output_filename, atoms_to_write.n_atoms) as W:
            for model_num, frame_index in enumerate(frame_indices):
                if 0 <= frame_index < len(universe.trajectory):
                    universe.trajectory[frame_index] # Go to frame
                    # Write with MODEL record implicitly handled by looping write calls in MDAnalysis >= 1.0 (?)
                    # Let's add explicit MODEL records just in case or for older versions
                    # W.write_timestep(ts=universe.trajectory.ts, model_num=model_num+1) # For older MDA?
                    W.write(atoms_to_write) # Write the selected atoms for the current frame
                else:
                    print(f"Warning: Frame index {frame_index} out of bounds ({len(universe.trajectory)} frames) for cluster {output_filename}. Skipping this frame.")
        if verbose:
            print(f"Saved SMD cluster ({len(frame_indices)} frames, indices {min(frame_indices)}-{max(frame_indices)}) to: {os.path.basename(output_filename)}")
        return True
    except Exception as e:
        print(f"Error extracting/saving frame cluster to {output_filename}: {e}")
        return False


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify metastable states from SMD/ProToken comparison and extract representative ProToken frames and corresponding SMD frame clusters."
    )
    # Inputs
    parser.add_argument("traj1", help="Path to Traj1 (SMD).")
    parser.add_argument("traj2", help="Path to Traj2 (ProToken).")
    parser.add_argument("start_pdb", help="Path to reference start PDB.")
    parser.add_argument("end_pdb", help="Path to reference end PDB.")
    parser.add_argument("selected_pairs_indices_pkl", help="Path to selected Cα pair indices (.pkl).")
    parser.add_argument("pca_model_pkl", help="Path to saved PCA model (.pkl).")
    parser.add_argument("--topology1", default=None, help="Optional topology for traj1.")
    parser.add_argument("--topology2", default=None, help="Optional topology for traj2.")
    # Params
    parser.add_argument("-s", "--atom_selection", default="protein and name CA", help="Atom selection for Cα distances/PCA.")
    parser.add_argument("--n_components_dtw", type=int, default=3, help="Num PCA components for DTW.")
    parser.add_argument("--min_smd_frames", type=int, default=10, help="Min SMD frames mapped.")
    parser.add_argument("--exclude_end_fraction", type=float, default=0.15, help="Exclude fraction of ProToken end.")
    parser.add_argument("--min_ratio", type=float, default=3.0, help="Min Path/Displacement ratio.")
    parser.add_argument("--peak_prominence", type=float, default=1.0, help="Required peak prominence.")
    # Output & Plotting (Defaults changed)
    parser.add_argument("-o", "--output_dir", default="metastable_cluster_analysis", help="Output directory.")
    parser.add_argument("--no_plot_pca", action="store_true", help="Disable PCA projection plot.")
    parser.add_argument("--no_plot_dtw", action="store_true", help="Disable DTW alignment plot.")
    parser.add_argument("--no_plot_ratios", action="store_true", help="Disable wandering ratio profile plot.")
    parser.add_argument("--extract_selection", default="protein", help="Atom selection for output PDB frames.")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity.")

    args = parser.parse_args()

    # Validate Args
    if not 0.0 <= args.exclude_end_fraction < 1.0: parser.error("--exclude_end_fraction must be >= 0.0 and < 1.0.")
    if args.n_components_dtw < 1: parser.error("--n_components_dtw must be >= 1.")

    try:
        # --- 0. Setup ---
        make_output_dir(args.output_dir)
        if args.verbose: print(f"Starting analysis. Outputs: {args.output_dir}")

        # --- 1. Load Models & Indices ---
        selected_pairs_indices = load_pickle(args.selected_pairs_indices_pkl, args.verbose)
        pca_model = load_pickle(args.pca_model_pkl, args.verbose)
        if not isinstance(selected_pairs_indices, list): raise TypeError("Loaded pairs not a list.")
        if not hasattr(pca_model, 'transform'): raise TypeError("Loaded PCA model invalid.")
        n_components_model = pca_model.n_components_
        if args.verbose: print(f"Loaded PCA model ({n_components_model} components), {len(selected_pairs_indices)} Cα pairs.")
        if args.n_components_dtw > n_components_model:
             print(f"Warning: Reducing n_components_dtw from {args.n_components_dtw} to {n_components_model} (model limit).")
             args.n_components_dtw = n_components_model

        # --- 2. Load Trajectories & Structures ---
        u1 = load_trajectory(args.traj1, args.topology1, args.verbose) # SMD
        u2 = load_trajectory(args.traj2, args.topology2, args.verbose) # ProToken
        ref_start_u, ref_start_atoms = load_static_structure(args.start_pdb, args.atom_selection, args.verbose)
        ref_end_u, ref_end_atoms = load_static_structure(args.end_pdb, args.atom_selection, args.verbose)

        # --- 3. Extract Features ---
        if args.verbose: print("\nExtracting features...")
        features1 = extract_trajectory_features(u1, selected_pairs_indices, args.atom_selection, args.verbose > 0)
        features2 = extract_trajectory_features(u2, selected_pairs_indices, args.atom_selection, args.verbose > 0)
        features_start = extract_static_features(ref_start_atoms, selected_pairs_indices)
        features_end = extract_static_features(ref_end_atoms, selected_pairs_indices)

        # --- 4. Project Features ---
        if args.verbose: print("\nProjecting features...")
        proj1 = pca_model.transform(features1)
        proj2 = pca_model.transform(features2)
        proj_start = pca_model.transform(features_start)
        proj_end = pca_model.transform(features_end)

        # --- 5. Plot PCA Projection (Default ON) ---
        pca_plot_path = os.path.join(args.output_dir, "pca_projection.png") if not args.no_plot_pca else None
        if pca_plot_path:
             plot_projected_paths(proj1, proj2, proj_start, proj_end, pca_plot_path,
                                title_suffix=f" ({len(selected_pairs_indices)} features)", verbose=args.verbose)

        # --- 6. Recalculate DTW & Plot (Default ON) ---
        dtw_plot_path = os.path.join(args.output_dir, "dtw_alignment.png") if not args.no_plot_dtw else None
        dtw_distance, best_path = calculate_and_plot_dtw(
            proj1[:, :args.n_components_dtw], proj2[:, :args.n_components_dtw],
            proj_start[:, :args.n_components_dtw], proj_end[:, :args.n_components_dtw],
            dtw_plot_path, verbose=args.verbose
        )
        if not best_path: print("\nDTW failed. Cannot proceed."); exit()
        print(f"\nDTW Distance (using {args.n_components_dtw} PCs): {dtw_distance:.4f}")

        # --- 7. Identify Metastable Candidates & Plot Ratios (Default ON) ---
        candidates, _ = find_metastable_candidates(proj1, proj2, best_path, args)

        # --- 8. Report and Extract Frames/Clusters ---
        print("\n--- Final Identified Potential Metastable States ---")
        if candidates:
            print("(ProToken_Frame, Num_SMD_Frames, SMD_Frame_Range, PathDispRatio_at_Peak)")
            candidate_info_lines = []
            # Create a dedicated subdirectory for extracted frames
            frames_output_dir = os.path.join(args.output_dir, "extracted_metastable_frames")
            make_output_dir(frames_output_dir)

            for candidate_num, (smd_indices, pro_idx, ratio_val) in enumerate(candidates):
                smd_range_str = f"{min(smd_indices)}-{max(smd_indices)}"
                print(f"Candidate {candidate_num+1}: ProTkn={pro_idx} <-> SMD Frames={len(smd_indices)} ({smd_range_str}), Ratio={ratio_val:.2f}")
                candidate_info_lines.append(f"{pro_idx}\t{len(smd_indices)}\t{smd_range_str}\t{ratio_val:.2f}\n")

                # Extract ProToken Frame
                pro_out_pdb = os.path.join(frames_output_dir, f"candidate_{candidate_num+1}_protoken_frame_{pro_idx}.pdb")
                if args.verbose: print(f"  Extracting ProToken frame {pro_idx}...")
                extract_and_save_frame(u2, pro_idx, pro_out_pdb, args.extract_selection, args.verbose > 1)

                # Extract SMD Cluster
                smd_cluster_out_pdb = os.path.join(frames_output_dir, f"candidate_{candidate_num+1}_smd_cluster_pro_{pro_idx}.pdb")
                if args.verbose: print(f"  Extracting SMD cluster ({len(smd_indices)} frames)...")
                extract_and_save_frame_cluster(u1, smd_indices, smd_cluster_out_pdb, args.extract_selection, args.verbose > 1)

            # Save candidates list to file
            candidates_file = os.path.join(args.output_dir, "metastable_candidates_summary.txt")
            with open(candidates_file, 'w') as f:
                 f.write("# Potential Metastable State Candidates\n")
                 f.write("# ProToken_Frame\tNum_SMD_Frames\tSMD_Frame_Range\tPathDispRatio_at_Peak\n")
                 f.writelines(candidate_info_lines)
            print(f"\nSaved candidates summary to: {candidates_file}")
            print(f"Extracted ProToken frames and SMD clusters saved in: {frames_output_dir}")

        else:
            print("No potential metastable states identified.")

    # Error Handling (same as before)
    except FileNotFoundError as fnf_err: print(f"\nError: Input file not found. {fnf_err}")
    except (TypeError, ValueError, IndexError) as data_err: print(f"\nError during processing: {data_err}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"\nAn unexpected error: {e}"); import traceback; traceback.print_exc()

    print("\nAnalysis complete.")
