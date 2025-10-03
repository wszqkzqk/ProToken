#!/usr/bin/env python3

import jax.numpy as jnp

def ca_rmsd(coords1, coords2, mask):
    """
    Calculates the C-alpha RMSD between two structures.

    Args:
        coords1: First set of atomic coordinates. Shape: (n_res, 37, 3)
        coords2: Second set of atomic coordinates. Shape: (n_res, 37, 3)
        mask: Sequence mask. Shape: (n_res,)

    Returns:
        Scalar RMSD value.
    """
    # Select C-alpha atoms (index 1)
    ca_coords1 = coords1[:, 1, :]
    ca_coords2 = coords2[:, 1, :]

    # Apply mask
    mask_expanded = mask[:, None]
    ca_coords1 = ca_coords1 * mask_expanded
    ca_coords2 = ca_coords2 * mask_expanded

    # Calculate squared differences
    squared_diff = jnp.sum((ca_coords1 - ca_coords2)**2, axis=-1)
    
    # Calculate mean squared difference, avoiding division by zero
    num_residues = jnp.sum(mask)
    mean_squared_diff = jnp.sum(squared_diff) / jnp.maximum(num_residues, 1e-8)

    return jnp.sqrt(mean_squared_diff)

def calculate_reward(coords_path, seq_mask):
    """
    Calculates the reward for a generated trajectory based on its smoothness.

    Args:
        coords_path: The decoded 3D coordinates trajectory. Shape: (n_points, n_res, 37, 3)
        seq_mask: The sequence mask for the protein. Shape: (n_res,)

    Returns:
        A single scalar reward value.
    """
    if coords_path.shape[0] < 2:
        return 0.0

    frame_to_frame_rmsds = []
    for i in range(coords_path.shape[0] - 1):
        rmsd = ca_rmsd(coords_path[i], coords_path[i+1], seq_mask)
        frame_to_frame_rmsds.append(rmsd)
    
    rmsds_array = jnp.array(frame_to_frame_rmsds)

    # Reward is the negative of the mean and max RMSD, to encourage smoothness
    mean_rmsd = jnp.mean(rmsds_array)
    max_rmsd = jnp.max(rmsds_array)

    # The negative sign turns the minimization of RMSD into the maximization of reward
    reward = - (mean_rmsd + max_rmsd)

    print(f"Trajectory evaluated. Mean RMSD: {mean_rmsd:.3f}, Max RMSD: {max_rmsd:.3f}, Reward: {reward:.3f}")

    return reward