#!/usr/bin/env python

import numpy as np
import pickle
import os
import argparse
from collections import defaultdict
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # For optional plotting

# --- Constants ---
EPSILON = 1e-6 # To avoid division by zero

# --- Helper Functions (reuse from previous script if needed) ---
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
    if len(coords) < 2:
        return 0.0, 0.0 # Cannot calculate for single point or empty

    # Path length: sum of distances between consecutive points
    diffs = np.diff(coords, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(step_distances)

    # Displacement: distance between start and end points
    displacement_vec = coords[-1] - coords[0]
    displacement = np.linalg.norm(displacement_vec)

    return path_length, displacement

# --- Main Logic ---

def find_metastable_candidates(proj1, proj2, best_path, args):
    """
    Identifies potential metastable states based on DTW alignment and SMD path analysis.

    Args:
        proj1 (np.ndarray): PCA projection of trajectory 1 (SMD), shape (n_frames_smd, n_components).
        proj2 (np.ndarray): PCA projection of trajectory 2 (ProToken), shape (n_frames_pro, n_components).
        best_path (list): List of (idx1, idx2) tuples from DTW alignment.
        args (argparse.Namespace): Command line arguments.

    Returns:
        list: List of tuples `(smd_representative_idx, protoken_peak_idx, ratio)` for identified candidates.
        np.ndarray: Array of calculated path/displacement ratios for each ProToken frame.
    """
    n_frames_smd = proj1.shape[0]
    n_frames_pro = proj2.shape[0]

    if args.verbose:
        print("\n--- Identifying Potential Metastable States ---")
        print(f"SMD frames: {n_frames_smd}, ProToken frames: {n_frames_pro}")

    # 1. Build ProToken to SMD mapping from DTW path
    pro_to_smd_map = defaultdict(list)
    for idx_smd, idx_pro in best_path:
        # Ensure indices are within bounds (sanity check)
        if 0 <= idx_smd < n_frames_smd and 0 <= idx_pro < n_frames_pro:
             pro_to_smd_map[idx_pro].append(idx_smd)
        else:
            if args.verbose:
                 print(f"Warning: DTW path index out of bounds: SMD={idx_smd}, ProToken={idx_pro}. Skipping.")


    if args.verbose:
        print("Built ProToken-to-SMD mapping from DTW path.")
        # Optional: print mapping stats
        # mapped_counts = {k: len(v) for k, v in pro_to_smd_map.items()}
        # print(f"Mapping counts per ProToken frame (min/max/avg): {min(mapped_counts.values())}/{max(mapped_counts.values())}/{np.mean(list(mapped_counts.values())):.2f}")


    # 2. Calculate Path/Displacement Ratio for non-terminal ProToken frames
    ratios = np.zeros(n_frames_pro) # Initialize ratios array
    end_frame_threshold_pro = int(n_frames_pro * (1 - args.exclude_end_fraction))

    print(f"Analyzing ProToken frames 0 to {end_frame_threshold_pro - 1} (excluding last {args.exclude_end_fraction*100:.0f}%).")

    for j in range(end_frame_threshold_pro): # Iterate through ProToken frames, excluding the end
        if j not in pro_to_smd_map:
            if args.verbose > 1: print(f"ProToken frame {j}: No SMD frames mapped. Ratio=0.")
            continue # Skip if no SMD frames mapped

        smd_indices = sorted(list(set(pro_to_smd_map[j]))) # Get unique sorted indices

        if len(smd_indices) < args.min_smd_frames:
            if args.verbose > 1: print(f"ProToken frame {j}: Mapped SMD frames ({len(smd_indices)}) < threshold ({args.min_smd_frames}). Ratio=0.")
            continue # Skip if not enough SMD frames

        # Select the corresponding PCA coordinates for the mapped SMD frames
        smd_segment_coords = proj1[smd_indices, :]

        # Calculate path length and displacement for this SMD segment
        path_length, displacement = calculate_path_metrics(smd_segment_coords)

        # Calculate ratio (handle near-zero displacement)
        ratio = path_length / (displacement + EPSILON)
        ratios[j] = ratio
        if args.verbose > 1:
             print(f"ProToken frame {j}: Mapped SMD frames={len(smd_indices)} ({min(smd_indices)}-{max(smd_indices)}), PathLen={path_length:.2f}, Disp={displacement:.2f}, Ratio={ratio:.2f}")


    # 3. Find peaks in the ratio profile
    # find_peaks requires the value to be strictly greater than neighbors
    # `prominence` helps filter out minor peaks, `height` sets a minimum ratio
    peaks, properties = find_peaks(ratios, height=args.min_ratio, prominence=args.peak_prominence)

    if args.verbose:
        print(f"\nFound {len(peaks)} potential peaks in Path/Displacement ratio profile meeting criteria:")
        if len(peaks) > 0:
             print(f"  Peak ProToken Indices: {peaks}")
             print(f"  Peak Ratios: {[f'{r:.2f}' for r in ratios[peaks]]}")
             if 'prominences' in properties:
                 print(f"  Peak Prominences: {[f'{p:.2f}' for p in properties['prominences']]}")
        else:
             print("  No peaks found meeting the criteria. Try adjusting thresholds (--min_ratio, --peak_prominence, --min_smd_frames, --exclude_end_fraction).")


    # 4. Extract Representative Frames for each peak
    metastable_candidates = []
    for j_peak in peaks:
        # ProToken representative is simply j_peak
        pro_rep_idx = j_peak

        # Find SMD representative
        smd_indices_peak = sorted(list(set(pro_to_smd_map[j_peak])))
        if not smd_indices_peak: # Should not happen if peak was found, but check
             if args.verbose: print(f"Warning: No SMD indices found for peak ProToken frame {j_peak}. Skipping.")
             continue

        # Use median SMD index as representative
        smd_rep_idx = smd_indices_peak[len(smd_indices_peak) // 2]

        metastable_candidates.append((smd_rep_idx, pro_rep_idx, ratios[j_peak]))
        if args.verbose:
             print(f"  Candidate: ProToken Frame={pro_rep_idx} (Ratio={ratios[j_peak]:.2f}) <-> Median SMD Frame={smd_rep_idx} (from range {min(smd_indices_peak)}-{max(smd_indices_peak)})")


    # 5. Optional: Plot the ratio profile
    if args.plot_ratios:
         fig, ax = plt.subplots(figsize=(10, 5))
         ax.plot(range(n_frames_pro), ratios, label='Path/Displacement Ratio', color='purple')
         ax.plot(peaks, ratios[peaks], "x", color='red', markersize=10, label='Identified Peaks')
         ax.axhline(args.min_ratio, color='gray', linestyle='--', label=f'Min Ratio Threshold ({args.min_ratio})')
         ax.set_xlabel("ProToken Frame Index")
         ax.set_ylabel("SMD Path Length / Displacement Ratio")
         ax.set_title("SMD Trajectory Wandering Metric per ProToken Frame")
         ax.legend()
         ax.grid(True, linestyle='--', alpha=0.6)
         ratio_plot_path = os.path.join(args.output_dir, "smd_wandering_ratio_profile.png")
         save_figure(fig, ratio_plot_path, args.verbose)


    return metastable_candidates, ratios

# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify potential metastable states by analyzing DTW alignment between two trajectories in PCA space."
    )
    # Input Files (previously generated by your other script)
    parser.add_argument("proj1_pca", help="Path to the PCA projected coordinates for Traj1 (SMD) (.pkl file).")
    parser.add_argument("proj2_pca", help="Path to the PCA projected coordinates for Traj2 (ProToken) (.pkl file).")
    parser.add_argument("dtw_path", help="Path to the best DTW alignment path (.pkl file).")
    # Parameters for Metastable State Identification
    parser.add_argument("--min_smd_frames", type=int, default=5,
                        help="Minimum number of SMD frames mapped to a ProToken frame to consider it.")
    parser.add_argument("--exclude_end_fraction", type=float, default=0.15,
                        help="Fraction of the ProToken trajectory end to exclude from analysis (e.g., 0.15 for last 15%).")
    parser.add_argument("--min_ratio", type=float, default=2.0,
                        help="Minimum Path/Displacement ratio threshold for a peak to be considered.")
    parser.add_argument("--peak_prominence", type=float, default=0.5,
                        help="Required prominence for peaks in the ratio profile (see scipy.signal.find_peaks).")
    # Output
    parser.add_argument("-o", "--output_dir", default="metastable_analysis", help="Directory to save outputs.")
    parser.add_argument("--plot_ratios", action="store_true", help="Plot the calculated ratio profile.")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity level (-v, -vv).")

    args = parser.parse_args()

    # Validate input fractions/percentages
    if not 0.0 <= args.exclude_end_fraction < 1.0:
        parser.error("--exclude_end_fraction must be between 0.0 and 1.0 (exclusive of 1.0).")

    try:
        # --- 0. Setup ---
        make_output_dir(args.output_dir)
        if args.verbose:
            print(f"Starting metastable state analysis. Outputs will be saved to: {args.output_dir}")

        # --- 1. Load Data ---
        # Note: Assuming your previous script saved these correctly
        proj1 = load_pickle(args.proj1_pca, args.verbose)
        proj2 = load_pickle(args.proj2_pca, args.verbose)
        # Assuming best_path is saved as a list of tuples
        best_path = load_pickle(args.dtw_path, args.verbose)

        # Basic checks on loaded data
        if not isinstance(proj1, np.ndarray) or proj1.ndim != 2:
             raise TypeError(f"Loaded proj1 data from {args.proj1_pca} is not a 2D numpy array.")
        if not isinstance(proj2, np.ndarray) or proj2.ndim != 2:
             raise TypeError(f"Loaded proj2 data from {args.proj2_pca} is not a 2D numpy array.")
        if not isinstance(best_path, list):
             # If dtaidistance saved paths object, extract best path
             if hasattr(best_path, 'get_best_path'): # Check if it's a Paths object
                 print("Detected dtaidistance Paths object, extracting best path...")
                 best_path = best_path.get_best_path()
             else:
                 raise TypeError(f"Loaded DTW path from {args.dtw_path} is not a list.")


        # --- 2. Find Candidates ---
        candidates, _ = find_metastable_candidates(proj1, proj2, best_path, args)

        # --- 3. Report Results ---
        print("\n--- Final Identified Potential Metastable State Pairs ---")
        if candidates:
            print("(SMD_Representative_Frame, ProToken_Frame, PathDispRatio_at_Peak)")
            for smd_idx, pro_idx, ratio_val in candidates:
                print(f"({smd_idx}, {pro_idx}, {ratio_val:.2f})")

            # Save candidates to file
            candidates_file = os.path.join(args.output_dir, "metastable_candidate_frames.txt")
            with open(candidates_file, 'w') as f:
                 f.write("# Potential Metastable State Representative Frames\n")
                 f.write("# SMD_Representative_Frame\tProToken_Frame\tPathDispRatio_at_Peak\n")
                 for smd_idx, pro_idx, ratio_val in candidates:
                      f.write(f"{smd_idx}\t{pro_idx}\t{ratio_val:.2f}\n")
            print(f"\nSaved candidate frame pairs to: {candidates_file}")
            print("\nUse these frame indices to extract corresponding PDB structures for visualization.")

        else:
            print("No potential metastable states identified with the current criteria.")


    except FileNotFoundError as fnf_err:
         print(f"\nError: Input file not found. {fnf_err}")
    except TypeError as te:
        print(f"\nError: Data type mismatch during loading or processing. {te}")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()

    print("\nMetastable analysis complete.")