import numpy as np
import pyemma
import os
import argparse

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="Comparing manually calculated C0 with PyEMMA's C0")
parser.add_argument('features',
                    help="Path to features .npy file")
parser.add_argument('tica',
                    help="Path to PyEMMA TICA object file (.pyemma)")
args = parser.parse_args()
features_filepath = args.features
pyemma_filepath = args.tica

print(f"Comparing manually calculated C0 with PyEMMA's C0")
print(f"Features file: {features_filepath}")
print(f"TICA object file: {pyemma_filepath}")

try:
    # Load data
    features_data = np.load(features_filepath).astype(np.float64) # Use float64 for precision
    tica_obj = pyemma.load(pyemma_filepath)
    pyemma_C0 = tica_obj.cov

    if pyemma_C0 is None:
            print("Error: PyEMMA TICA object does not contain C0 (.cov).")
            exit()
    if features_data.shape[1] != pyemma_C0.shape[0]:
            print("Error: Feature dimension mismatch between data file and TICA object.")
            exit()

    print(f"Features shape: {features_data.shape}")
    print(f"PyEMMA C0 shape: {pyemma_C0.shape}")

    # Manually calculate C0 using numpy.cov
    # Note: np.cov expects features in ROWS, so transpose features_data
    # It automatically handles mean subtraction.
    print("\nCalculating C0 manually using np.cov...")
    manual_C0 = np.cov(features_data, rowvar=False, bias=True)
    # Use bias=True to match population covariance (divide by N),
    # which is closer to what TICA often uses internally than sample covariance (N-1).
    # If reversible=True was used in TICA, PyEMMA's C0 might still differ slightly due to averaging over pairs.

    print(f"Manual C0 shape: {manual_C0.shape}")

    # Compare the matrices
    # Use np.allclose for element-wise comparison with tolerance
    tolerance = 1e-6 # Adjust tolerance as needed
    are_close = np.allclose(manual_C0, pyemma_C0, atol=tolerance)

    print(f"\nAre manual C0 and PyEMMA C0 close (tolerance={tolerance:.1e})? {are_close}")

    if not are_close:
        print("Significant difference found between manual C0 and PyEMMA C0.")
        # Optional: Calculate and print difference norm
        diff_norm = np.linalg.norm(manual_C0 - pyemma_C0)
        relative_diff = diff_norm / np.linalg.norm(pyemma_C0) if np.linalg.norm(pyemma_C0) > 0 else diff_norm
        print(f"Norm of difference: {diff_norm:.4e}")
        print(f"Relative difference: {relative_diff:.4e}")
        print("Possible reasons: different mean/covariance calculation (e.g., bias, weights), or PyEMMA's symmetric averaging if reversible=True was used.")

except FileNotFoundError:
    print("Error: Could not find features or TICA object file.")
except Exception as e:
    print(f"An error occurred during comparison: {e}")

