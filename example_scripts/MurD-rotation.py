import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer
from scipy.spatial.transform import Rotation as R
import argparse
import os

# Set chain ID and residue ranges for domains
CHAIN_ID = 'A'
FIXED_DOMAIN_RANGE = (1, 299)    # N-terminal + central domain
MOVING_DOMAIN_RANGE = (300, 437) # C-terminal domain

def get_domain_coords(model, chain_id, res_range):
    """
    Extract CA atoms for residues in the specified range from the model,
    returning a list of atoms.
    """
    atoms = []
    chain = model[chain_id]
    for residue in chain:
        res_id = residue.get_id()[1]
        if res_range[0] <= res_id <= res_range[1]:
            if 'CA' in residue:
                atoms.append(residue['CA'])
    return atoms

def compute_domain_transformation(ref_atoms, target_atoms):
    """
    Use the Kabsch algorithm (Bio.PDB.Superimposer) to compute the rotation matrix and translation vector 
    from target_coords to ref_coords.
    Returns the rotation matrix (3x3) and the translation vector (3,).
    """
    si = Superimposer()
    si.set_atoms(ref_atoms, target_atoms)
    rot, trans = si.rotran  # rot: 3x3 matrix, trans: translation vector
    return rot, trans

def extract_rotation_angle_translation(rot, trans):
    """
    Use scipy to compute the rotation angle (in radians) and rotation axis.
    Also, compute the projection of the translation vector along the rotation axis.
    """
    r = R.from_matrix(rot)
    angle = r.magnitude()  # rotation angle in radians
    # Get rotation axis (set as zero vector if the angle is negligible)
    axis = r.as_rotvec()
    if angle > 1e-6:
        axis = axis / angle
    else:
        axis = np.array([0, 0, 0])
    # Component of the translation along the rotation axis
    trans_along_axis = np.dot(trans, axis)
    return angle, trans_along_axis, axis

def process_multiframe_pdb(pdb_file, chain_id, fixed_range, moving_range):
    """
    Read a multi-model PDB file using the first model as reference.
    For each frame, compute the alignment transformation for the fixed domain (N-terminal + central),
    then compute the relative transformation for the moving domain (C-terminal).
    Returns the rotation angle (in radians), translation component (along rotation axis),
    and the rotation axis for each frame of the C-terminal domain.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("murD", pdb_file)
    
    models = list(structure)
    # Use the first model as the reference model
    ref_model = models[0]
    ref_fixed = get_domain_coords(ref_model, chain_id, fixed_range)
    ref_moving = get_domain_coords(ref_model, chain_id, moving_range)
    
    results = []
    
    # Process each model/frame
    for i, model in enumerate(models):
        # Extract coordinates for fixed and moving domains
        target_fixed = get_domain_coords(model, chain_id, fixed_range)
        target_moving = get_domain_coords(model, chain_id, moving_range)
        
        # Compute alignment transformation for the fixed domain (align target_fixed to ref_fixed)
        rot_fixed, trans_fixed = compute_domain_transformation(ref_fixed, target_fixed)

        # Apply the same transformation to the target model's moving domain coordinates
        # Get the coordinates of the moving atoms
        target_moving_coords = np.array([atom.get_coord() for atom in target_moving])
        # Apply the rotation and translation
        target_moving_aligned = np.dot(target_moving_coords, rot_fixed.T) + trans_fixed

        # Create a list of dummy atoms with the aligned coordinates
        aligned_atoms = []
        for j, atom in enumerate(target_moving):
            dummy_atom = atom.copy()
            dummy_atom.set_coord(target_moving_aligned[j])
            aligned_atoms.append(dummy_atom)
        
        # Compute the optimal transformation between ref_moving and aligned target_moving of the moving domain
        rot_moving, trans_moving = compute_domain_transformation(ref_moving, aligned_atoms)
        
        # Extract rotation angle, translation along axis, and rotation axis
        angle, trans_along_axis, axis = extract_rotation_angle_translation(rot_moving, trans_moving)
        results.append({
            "model_index": i,
            "rotation_angle_radians": angle,
            "rotation_angle_degrees": np.degrees(angle),
            "translation_along_axis": trans_along_axis,
            "rotation_axis": axis
        })
    return results

def plot_results(results, output_path):
    """
    Plot rotation angles and translation along axis vs. model index.
    Save the plot to the specified output path.
    """
    model_indices = [res["model_index"] for res in results]
    rotation_angles = [res["rotation_angle_degrees"] for res in results]
    translations = [res["translation_along_axis"] for res in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Model Index')
    ax1.set_ylabel('Rotation Angle (°)', color=color)
    ax1.plot(model_indices, rotation_angles, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Translation Along Axis (Å)', color=color)  # we already handled the x-label with ax1
    ax2.plot(model_indices, translations, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Rotation and Translation vs. Model Index')
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze domain rotations and translations in a multi-frame PDB file.")
    parser.add_argument("pdb_file", help="Path to the input multi-frame PDB file.")
    parser.add_argument("-o", "--output_file", default="results_plot.svg", help="Path to the output image file (default: results_plot.svg).")
    args = parser.parse_args()
    pdb_filename = args.pdb_file
    output_file = args.output_file

    results = process_multiframe_pdb(pdb_filename, CHAIN_ID, FIXED_DOMAIN_RANGE, MOVING_DOMAIN_RANGE)
    for res in results:
        print("Model {}: Rotation angle {:.2f}°, translation along axis {:.3f} Å, rotation axis {}"
              .format(res["model_index"],
                      res["rotation_angle_degrees"],
                      res["translation_along_axis"],
                      res["rotation_axis"]))

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_results(results, output_file)
    print(f"Plot saved to {output_file}")
