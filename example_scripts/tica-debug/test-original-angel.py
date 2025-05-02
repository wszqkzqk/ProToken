import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Process torsion angles and plot histograms.')
parser.add_argument('data_path', help='Path to the .npz data file')
args = parser.parse_args()
data_path = args.data_path

data_dir = os.path.dirname(data_path)
data = np.load(data_path)
phi = data["phi"]; psi = data["psi"]; omega = data["omega"]

print("phi shape:", phi.shape, " NaN:", np.isnan(phi).sum(), " Inf:", np.isinf(phi).sum())
print("psi shape:", psi.shape, " NaN:", np.isnan(psi).sum(), " Inf:", np.isinf(psi).sum())
print("omega shape:", omega.shape, " NaN:", np.isnan(omega).sum(), " Inf:", np.isinf(omega).sum())

for name, arr in (("phi",phi),("psi",psi),("omega",omega)):
    print(f"{name} min/max = {arr.min():.3f}/{arr.max():.3f}  ([-π, π])")

out_fig = os.path.join(data_dir, "raw_angles_hist.png")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, arr) in zip(axes, (("phi", phi), ("psi", psi), ("omega", omega))):
    ax.hist(arr.flatten(), bins=100, range=(-np.pi, np.pi), color="C0", alpha=0.7)
    rug_y = np.full_like(data, -0.02)
    ax.plot(data, rug_y, '|', color='k', alpha=0.6, markersize=4)
    ax.set_title(name)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_fig, dpi=300)
print(f"Saved to {os.path.abspath(out_fig)}")

# (Phi vs Psi)
ramachandran_fig = os.path.join(data_dir, "ramachandran_plot.png")
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.scatter(phi.flatten(), psi.flatten(), s=1, alpha=0.3, color="C1")
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-np.pi, np.pi)
ax2.set_xlabel(r'$\phi$')
ax2.set_ylabel(r'$\psi$')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
ax2.set_xticks(ticks); ax2.set_xticklabels(labels)
ax2.set_yticks(ticks); ax2.set_yticklabels(labels)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ramachandran_fig, dpi=300)
print(f"Saved to {os.path.abspath(ramachandran_fig)}")
