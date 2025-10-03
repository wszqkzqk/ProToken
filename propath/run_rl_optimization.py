#!/usr/bin/env python3

import argparse
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import pickle as pkl

# --- Add project root to sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports from our modules ---
from propath.environment import PathDecodingEnvironment
from propath.policy_network import PolicyNetwork
from propath.reward import calculate_reward

# --- Imports from the original project structure ---
from run_optimization import load_models, encode_structure # Re-using the functions we built
from data.protein_utils import save_pdb_from_aux
from jax.tree_util import tree_map


def parse_rl_arguments():
    parser = argparse.ArgumentParser(description="Run RL-based optimization for protein transition paths.")
    parser.add_argument("--pdb_start", type=str, required=True, help="Path to the starting PDB file.")
    parser.add_argument("--pdb_end", type=str, required=True, help="Path to the ending PDB file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--n_episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the policy optimizer.")
    parser.add_argument("--n_path_points", type=int, default=50, help="Number of points in the transition path.")
    parser.add_argument("--exploration_std_dev", type=float, default=0.1, help="Standard deviation for exploration noise.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def save_trajectory_as_pdb(coords_path, aatype, residue_index, seq_mask, file_path):
    """Saves a trajectory of coordinates as a multi-model PDB file."""
    print(f"Saving trajectory to {file_path}...")
    with open(file_path, 'w') as f:
        for i, coords in enumerate(coords_path):
            f.write(f"MODEL        {i+1}\n")
            aux_data = {
                "aatype": aatype,
                "residue_index": residue_index,
                "atom_positions": coords,
                "atom_mask": jnp.ones_like(coords) * seq_mask[:, None, None],
                "plddt": jnp.ones_like(seq_mask) * 1.0
            }
            # save_pdb_from_aux expects numpy arrays
            aux_data_np = tree_map(lambda x: np.asarray(x), aux_data)
            pdb_str = save_pdb_from_aux(aux_data_np, return_string=True)
            f.write(pdb_str)
            f.write("ENDMDL\n")
    print("Save complete.")


def main():
    args = parse_rl_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Initialization ---
    key = jax.random.PRNGKey(args.seed)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("1. Loading all models...")
    models = load_models(project_root)
    
    print("2. Initializing Environment and Policy Network...")
    env = PathDecodingEnvironment(models)
    
    # Get latent space dimensionality from the loaded models
    latent_dim = models['protoken_emb'].shape[-1] + models['aatype_emb'].shape[-1]
    n_res = 512 # TODO: Get this from config

    policy = PolicyNetwork(n_points=args.n_path_points, n_res=n_res, latent_dim=latent_dim)

    print("3. Encoding start and end PDBs...")
    # We need aatype and residue_index for the environment and saving results
    _, _, native_seq_len = protoken_basic_generator(args.pdb_start, NUM_RES=n_res)
    native_seq_mask = jnp.pad(jnp.ones(native_seq_len), (0, n_res - native_seq_len))
    # This is a simplification; a more robust way would be to parse the PDB properly
    from data.dataset import protoken_basic_generator
    native_features, _, _ = protoken_basic_generator(args.pdb_start, NUM_RES=n_res)
    native_aatype = native_features['aatype']
    native_residue_index = native_features['residue_index']

    xT_start = encode_structure(args.pdb_start, models, args)
    xT_end = encode_structure(args.pdb_end, models, args)

    # Initialize policy network parameters
    key, policy_key = jax.random.split(key)
    policy_params = policy.init(policy_key, xT_start, xT_end, key)['params']
    
    # Initialize optimizer and training state
    optimizer = optax.adam(learning_rate=args.learning_rate)
    state = train_state.TrainState.create(apply_fn=policy.apply, params=policy_params, tx=optimizer)

    best_reward = -jnp.inf
    best_path = None

    # --- RL Training Loop (REINFORCE) ---
    print("\n--- Starting RL Training ---")
    for episode in range(args.n_episodes):
        key, episode_key = jax.random.split(key)

        # Define the loss function for a single episode
        def loss_fn(params):
            # 1. Sample an action (a latent path) from the policy
            xT_path, log_prob = policy.apply({'params': params}, xT_start, xT_end, episode_key, args.exploration_std_dev)
            
            # 2. Execute the action in the environment to get the 3D trajectory
            coords_path = env.decode_path_to_coords(xT_path, native_seq_mask, native_residue_index, native_aatype)

            # 3. Calculate the reward
            reward = calculate_reward(coords_path, native_seq_mask)

            # 4. Calculate Policy Gradient loss
            # We want to maximize reward, so we minimize -reward * log_prob
            loss = -reward * log_prob
            return loss, (reward, xT_path, coords_path)

        # --- Update Step ---
        (loss, (reward, xT_path, coords_path)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        print(f"Episode {episode+1}/{args.n_episodes} -> Loss: {loss:.2f}, Reward: {reward:.3f}")

        # --- Save best results ---
        if reward > best_reward:
            print(f"  ** New best reward! Saving results. **")
            best_reward = reward
            best_path = coords_path
            
            # Save the best PDB trajectory
            pdb_path = os.path.join(args.output_dir, "best_trajectory.pdb")
            save_trajectory_as_pdb(best_path, native_aatype, native_residue_index, native_seq_mask, pdb_path)
            
            # Save the latent path that generated it
            latent_path_file = os.path.join(args.output_dir, "best_latent_path.pkl")
            with open(latent_path_file, 'wb') as f:
                pkl.dump(xT_path, f)

    print("\n--- RL Training Finished ---")
    print(f"Best reward achieved: {best_reward:.3f}")
    print(f"Final results saved in: {args.output_dir}")

if __name__ == "__main__":
    # This import is here because it might initialize JAX on multiple devices, 
    # which can be slow. It's better to have it inside main.
    from data.dataset import protoken_basic_generator
    main()