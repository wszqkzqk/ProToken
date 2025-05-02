#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for saving files without showing
import matplotlib.pyplot as plt

# Suppress PyEMMA/MDTraj warnings if desired
# warnings.filterwarnings('ignore', category=UserWarning)

# --- Main Analysis Function ---
def analyze_and_visualize_c0(pyemma_filepath, output_dir, plot_format='png'):
    """
    Loads a PyEMMA TICA object, analyzes its C0 matrix, and saves visualizations.

    Args:
        pyemma_filepath (str): Path to the input .pyemma TICA model file.
        output_dir (str): Directory to save the output plots.
        plot_format (str): Format for the saved plots (e.g., 'png', 'pdf', 'svg').
    """
    print(f"--- Analyzing C0 for: {pyemma_filepath} ---")

    # --- Load TICA Object ---
    try:
        # Dynamically import pyemma here to allow script execution even if it's missing initially
        # (though it won't work without it)
        import pyemma
        print(f"Loading TICA object...")
        tica_obj = pyemma.load(pyemma_filepath)
        print("TICA object loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{pyemma_filepath}'", file=sys.stderr)
        sys.exit(1)
    except ImportError:
         print("Error: PyEMMA library not found. Please install PyEMMA.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"Error loading PyEMMA file '{pyemma_filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Access and Validate C0 Matrix ---
    try:
        C0 = tica_obj.cov
        if C0 is None:
            print("Error: C0 matrix (tica_obj.cov) is None. Estimation might have failed.", file=sys.stderr)
            sys.exit(1)
        print(f"C0 shape: {C0.shape}")

        if np.isnan(C0).any() or np.isinf(C0).any():
            print("Error: C0 matrix contains NaN or Inf values! Cannot proceed.", file=sys.stderr)
            sys.exit(1)

    except AttributeError:
        print(f"Error: Loaded object from '{pyemma_filepath}' does not have a '.cov' attribute. Is it a valid TICA object?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing C0 matrix: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Prepare Output Filenames ---
    base_input_name = os.path.splitext(os.path.basename(pyemma_filepath))[0]
    heatmap_filename = f"{base_input_name}_C0_heatmap.{plot_format}"
    spectrum_filename = f"{base_input_name}_C0_spectrum.{plot_format}"
    heatmap_filepath = os.path.join(output_dir, heatmap_filename)
    spectrum_filepath = os.path.join(output_dir, spectrum_filename)


    # --- Visualization 1: Heatmap of C0 ---
    try:
        print(f"\n--- Generating C0 Heatmap ({heatmap_filename}) ---")
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 7))
        cmap = 'coolwarm'
        abs_max = np.max(np.abs(C0)) if C0.size > 0 else 1.0
        clim_val = max(abs_max, 1e-9)

        im = ax_heatmap.imshow(C0, cmap=cmap, vmin=-clim_val, vmax=clim_val, aspect='auto')
        fig_heatmap.colorbar(im, ax=ax_heatmap, label='Covariance')
        ax_heatmap.set_title(f'C0 (Instantaneous Covariance Matrix)\nSource: {os.path.basename(pyemma_filepath)}')
        ax_heatmap.set_xlabel("Feature Index")
        ax_heatmap.set_ylabel("Feature Index")
        fig_heatmap.tight_layout()
        plt.savefig(heatmap_filepath, dpi=300)
        plt.close(fig_heatmap) # Close figure to free memory
        print(f"Heatmap saved to: {heatmap_filepath}")

    except Exception as e:
        print(f"Error generating/saving heatmap: {e}", file=sys.stderr)


    # --- Numerical Checks for C0 ---
    print("\n--- Numerical Checks for C0 ---")
    try:
        is_symmetric = np.allclose(C0, C0.T, atol=1e-8)
        print(f"Is C0 numerically symmetric? {is_symmetric}")
        if not is_symmetric:
            print("Warning: C0 is not symmetric, which is unexpected for covariance.")

        print("Calculating eigenvalues of C0...")
        eigvals_C0 = np.linalg.eigvalsh(C0) # Use eigvalsh since C0 should be symmetric

        min_eig = np.min(eigvals_C0)
        max_eig = np.max(eigvals_C0)
        print(f"Min eigenvalue of C0: {min_eig:.4e}")
        print(f"Max eigenvalue of C0: {max_eig:.4e}")

        num_negative = np.sum(eigvals_C0 < -1e-10)
        if num_negative > 0:
            print(f"WARNING: Found {num_negative} eigenvalues significantly < 0!")

        zero_threshold = 1e-8
        num_near_zero = np.sum(np.abs(eigvals_C0) < zero_threshold)
        print(f"Number of near-zero eigenvalues (< {zero_threshold:.1e}): {num_near_zero} / {len(eigvals_C0)}")

    except np.linalg.LinAlgError as e:
        print(f"Error during eigenvalue calculation: {e}. Matrix might be ill-formed.", file=sys.stderr)
        eigvals_C0 = None # Prevent further analysis based on eigenvalues
    except Exception as e:
        print(f"Error during numerical checks: {e}", file=sys.stderr)
        eigvals_C0 = None

    # --- Visualization 2: Eigenvalue Spectrum ---
    if eigvals_C0 is not None and len(eigvals_C0) > 0:
        try:
            print(f"\n--- Generating C0 Eigenvalue Spectrum ({spectrum_filename}) ---")
            fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 5))

            sorted_eigvals = np.sort(eigvals_C0)[::-1] # Sort descending
            positive_eigvals = sorted_eigvals[sorted_eigvals > zero_threshold]
            indices = np.arange(1, len(positive_eigvals) + 1)

            if len(positive_eigvals) > 0:
                ax_spectrum.semilogy(indices, positive_eigvals, marker='.', linestyle='-', markersize=4)
                ax_spectrum.set_title("C0 Eigenvalue Spectrum (Sorted Descending)")
                ax_spectrum.set_xlabel("Eigenvalue Index")
                ax_spectrum.set_ylabel("Eigenvalue Magnitude (log scale)")
                ax_spectrum.grid(True, which='both', linestyle='--', alpha=0.6)
                ax_spectrum.set_xlim(left=0.8)
                print(f"Plotting {len(positive_eigvals)} positive eigenvalues (> {zero_threshold:.1e}).")
                if len(positive_eigvals) < len(sorted_eigvals):
                    print(f"Excluded {len(sorted_eigvals) - len(positive_eigvals)} non-positive/near-zero eigenvalues.")

                # Calculate and print condition number using plotted values
                min_plotted_eigval = positive_eigvals[-1]
                max_plotted_eigval = positive_eigvals[0]
                condition_number_C0 = max_plotted_eigval / min_plotted_eigval
                print(f"Condition number of C0 (using plotted positive eigenvalues): {condition_number_C0:.4e}")
                if condition_number_C0 > 1e8:
                    print("Warning: Condition number is large -> ill-conditioned.")

            else:
                print("No positive eigenvalues > threshold found to plot on log scale.")
                ax_spectrum.text(0.5, 0.5, 'No positive eigenvalues > threshold found',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax_spectrum.transAxes)

            fig_spectrum.tight_layout()
            plt.savefig(spectrum_filepath, dpi=300)
            plt.close(fig_spectrum) # Close figure
            print(f"Spectrum plot saved to: {spectrum_filepath}")

        except Exception as e:
            print(f"Error generating/saving spectrum plot: {e}", file=sys.stderr)

    print(f"\n--- Analysis complete for: {pyemma_filepath} ---")


# --- CLI Argument Parser ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize the C0 (instantaneous covariance) matrix '
                    'from a saved PyEMMA TICA object file (.pyemma).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_pyemma_file',
        type=str,
        help='Path to the input PyEMMA TICA model file (.pyemma).'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Directory to save the output plots (heatmap and spectrum). '
             'Defaults to the directory containing the input file.'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        help='Image format for the output plots (e.g., png, pdf, svg, jpg).'
    )
    return parser.parse_args()


# --- Main Execution Block ---
if __name__ == "__main__":
    args = parse_arguments()

    # Validate input file path
    input_filepath = os.path.abspath(args.input_pyemma_file)
    if not os.path.isfile(input_filepath):
        print(f"Error: Input file not found: {input_filepath}", file=sys.stderr)
        sys.exit(1)

    # Determine and create output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(input_filepath)
    else:
        output_dir = args.output_dir

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
    except OSError as e:
        print(f"Error: Cannot create output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # Run the analysis
    analyze_and_visualize_c0(input_filepath, output_dir, args.format)
