import numpy as np
import pandas as pd  # using pandas for duplicate column detection
import argparse

parser = argparse.ArgumentParser(description="Check TICA feature .npy file for low-variance and duplicate columns")
parser.add_argument(
    "features_filepath",
    help="Path to the feature .npy file to analyze"
)
args = parser.parse_args()
features_filepath = args.features_filepath

print(f"Checking features file: {features_filepath}")
features_data = np.load(features_filepath)
print(f"Features shape: {features_data.shape}")  # e.g., (1001, 1620)

# Check 1: Constant columns (very low variance)
variances = np.var(features_data, axis=0)
low_var_threshold = 1e-10  # Use a very small threshold
const_cols_indices = np.where(variances < low_var_threshold)[0]
print(f"\nNumber of columns with variance < {low_var_threshold:.1e}: {len(const_cols_indices)}")
if len(const_cols_indices) > 0:
    print(f"Indices of potentially constant columns: {const_cols_indices[:20]}...")  # Show first few

# Check 2: Duplicate columns
print("\nChecking for duplicate columns (this might take a moment)...")
df_features = pd.DataFrame(features_data)
# T attribute transposes so columns become rows for duplication check
duplicates_series = df_features.T.duplicated(keep=False)  # Mark all duplicates
duplicate_cols_indices = np.where(duplicates_series)[0]
n_duplicates = len(duplicate_cols_indices)
# Calculate how many *unique* duplicate sets exist
unique_duplicate_count = 0
if n_duplicates > 0:
        # Count how many columns are duplicates (excluding the first occurrence of each set)
        n_redundant_cols = df_features.T.duplicated(keep='first').sum()
        unique_duplicate_count = n_duplicates - n_redundant_cols
        print(f"Found {n_redundant_cols} redundant columns (duplicates of earlier columns).")
        print(f"Total columns involved in duplication groups: {n_duplicates}")
        print(f"Indices of columns involved in duplicates: {duplicate_cols_indices[:20]}...")  # Show first few
else:
        print("No duplicate columns found.")

total_problematic_cols = len(const_cols_indices) + (n_duplicates - unique_duplicate_count if n_duplicates > 0 else 0)
print(f"\nEstimated number of dimensions lost due to constant/duplicate columns: >= {total_problematic_cols}")
print(f"(Compare this to the {620} near-zero eigenvalues found previously)")

