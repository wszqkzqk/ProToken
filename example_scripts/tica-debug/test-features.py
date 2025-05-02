#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Analyze features .npy file for debugging')
parser.add_argument('features-file', required=True, help='Path to the .npy features file')
args = parser.parse_args()

features_file = args.features_file
features_dir = os.path.dirname(features_file)
features = np.load(features_file)

print(f"Loaded features shape: {features.shape}")
# Expected output: Loaded features shape: (1000, N) where N is the number of features

# Check for NaN or Inf values
if np.isnan(features).any() or np.isinf(features).any():
    print("Error: Features contain NaN or Inf values!")
    # Find specific indices of NaN/Inf
    nan_indices = np.where(np.isnan(features))
    inf_indices = np.where(np.isinf(features))
    # Show first 10 occurrences
    print(f"NaN locations (frame, feature_index): {list(zip(nan_indices[0], nan_indices[1]))[:10]}")
    print(f"Inf locations (frame, feature_index): {list(zip(inf_indices[0], inf_indices[1]))[:10]}")
else:
    print("No NaN or Inf values found in features.")

# Report the range of feature values
print(f"Feature value range: min={np.min(features):.4f}, max={np.max(features):.4f}")
# Warn if values are outside the expected [-1, 1] range
if np.min(features) < -1.01 or np.max(features) > 1.01:
     print("Warning: Feature values outside expected [-1, 1] range for sin/cos!")

# Plot some features vs. frame index
num_frames = features.shape[0]
num_features_to_plot = min(6, features.shape[1]) # only plot the first few features
fig, axes = plt.subplots(num_features_to_plot, 1, figsize=(10, 2 * num_features_to_plot), sharex=True)
# Handle single subplot case
if num_features_to_plot == 1: axes = [axes]
for i in range(num_features_to_plot):
    axes[i].plot(np.arange(num_frames), features[:, i], marker='.', linestyle='-', markersize=2)
    axes[i].set_ylabel(f'Feature {i}')
    axes[i].grid(True, alpha=0.5)
axes[-1].set_xlabel('Frame Index')
fig.suptitle('Feature Values vs. Frame Index (First Few Features)')
# Adjust layout to prevent title overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
# Save and close
plt.savefig(os.path.join(features_dir, "debug_feature_vs_time.png"))
plt.close(fig)
print("Saved plot: debug_feature_vs_time.png")

# Check variance: detect features that barely change
variances = np.var(features, axis=0)
print(f"Variance range: min={np.min(variances):.4g}, max={np.max(variances):.4g}")
# set a threshold
if np.min(variances) < 1e-6:
    low_variance_indices = np.where(variances < 1e-6)[0]
    print(f"Warning: Features with very low variance detected at indices: {low_variance_indices}")
    # low variance features may cause unstable TICA or weird results
