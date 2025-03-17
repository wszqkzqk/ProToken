#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute centroids for LIDbd (residues 118-167), NMPbd (residues 30-67) and CORE (residues 1-29), then plot distances between CORE and the other domains."
    )
    parser.add_argument("pdb_file", help="Input multi-model PDB file")
    parser.add_argument("-c", "--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("-o", "--output", default="domain_distance.png", help="Output image file (png or svg)")
    return parser.parse_args()

def compute_centroid(residues):
    """
    Compute the centroid of a list of residues by averaging the coordinates of all atoms.
    """
    coords = []
    for residue in residues:
        for atom in residue:
            coords.append(atom.get_coord())
    if coords:
        coords = np.array(coords)
        centroid = np.mean(coords, axis=0)
        return centroid
    else:
        return None

def get_residues_by_range(chain, start, end):
    """
    Retrieve residues from the given chain with residue numbers between start and end (inclusive).
    """
    residues = []
    for residue in chain:
        # residue.get_id() returns a tuple; the residue number is at index 1.
        resnum = residue.get_id()[1]
        if start <= resnum <= end:
            residues.append(residue)
    return residues

def compute_domain_distances(pdb_file, chain_id):
    """
    For each MODEL in the PDB file, compute centroids for:
      - CORE domain: residues 1 to 29, 68 to 117, and 161 to 214
      - NMPbd domain: residues 30-67
      - LIDbd domain: residues 118-167
    Then calculate the distances between CORE and NMPbd, and between CORE and LIDbd.
    Returns lists of model IDs, CORE-LID distances, and CORE-NMP distances.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AdK", pdb_file)
    model_ids = []
    core_lid_distances = []
    core_nmp_distances = []

    for model in structure:
        try:
            chain = model[chain_id]
        except KeyError:
            print(f"Model {model.id} does not contain chain {chain_id}. Skipping.")
            continue

        # Get residues for each domain
        # CORE domain now consists of three segments
        core_residues1 = get_residues_by_range(chain, 1, 29)
        core_residues2 = get_residues_by_range(chain, 68, 117)
        core_residues3 = get_residues_by_range(chain, 161, 214)
        core_residues = core_residues1 + core_residues2 + core_residues3
        
        nmp_residues = get_residues_by_range(chain, 30, 67)
        lid_residues = get_residues_by_range(chain, 118, 167)

        # Compute centroids
        core_centroid = compute_centroid(core_residues)
        nmp_centroid = compute_centroid(nmp_residues)
        lid_centroid = compute_centroid(lid_residues)

        if core_centroid is None:
            print(f"Model {model.id} is missing CORE domain data. Skipping.")
            continue
        if nmp_centroid is None:
            print(f"Model {model.id} is missing NMPbd domain data. Skipping.")
            continue
        if lid_centroid is None:
            print(f"Model {model.id} is missing LIDbd domain data. Skipping.")
            continue

        # Compute distances between CORE centroid and the centroids of LIDbd and NMPbd
        distance_core_nmp = np.linalg.norm(core_centroid - nmp_centroid)
        distance_core_lid = np.linalg.norm(core_centroid - lid_centroid)

        model_ids.append(model.id)
        core_nmp_distances.append(distance_core_nmp)
        core_lid_distances.append(distance_core_lid)

    return model_ids, core_lid_distances, core_nmp_distances

def plot_domain_distances(model_ids, core_lid, core_nmp, output_file):
    """
    Plot the distances between CORE and LIDbd, and CORE and NMPbd across MODEL frames.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(model_ids, core_lid, marker='o', linestyle='-', color='blue', label="CORE-LID Distance")
    plt.plot(model_ids, core_nmp, marker='s', linestyle='-', color='red', label="CORE-NMP Distance")
    plt.xlabel("Model Frame Number")
    plt.ylabel("Distance (Å)")
    plt.title("Domain Distance Changes During Closure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

def main():
    args = parse_args()
    model_ids, core_lid, core_nmp = compute_domain_distances(args.pdb_file, args.chain)
    if not model_ids:
        print("No valid model data found. Exiting.")
        return
    plot_domain_distances(model_ids, core_lid, core_nmp, args.output)

if __name__ == "__main__":
    main()
