
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
sys.path.append('..')

import jax.numpy as jnp
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import our custom modules
from propath.rl_path_finder import ProTokenWrapper, PathEvaluator
# This import is needed for the initial path generation
from PROTOKEN.data.dataset import protoken_basic_generator

def evaluate_path_energy(path, evaluator, end_coords, wrapper, feature):
    """Calculates the total 'energy' of a given path."""
    total_energy = 0.0
    
    # Decode all intermediate structures once
    decoded_coords = []
    for emb in path:
        coords = wrapper.decode(emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], "/dev/null")
        decoded_coords.append(coords)

    for i in range(len(path) - 1):
        # Smoothness penalty
        step_rmsd = evaluator.calculate_rmsd(decoded_coords[i], decoded_coords[i+1])
        total_energy += step_rmsd # Higher RMSD = higher energy

        # Progress penalty (distance to target)
        # We want to encourage later steps to be closer to the end
        dist_to_target = evaluator.calculate_rmsd(decoded_coords[i+1], end_coords)
        total_energy += dist_to_target * 0.5 # Add a weight to this term
        
    return total_energy

def metropolis_search(args):
    """Performs a path search using Metropolis Monte Carlo with Simulated Annealing."""

    # 1. Initialization
    print("Initializing ProToken wrapper, evaluator, and initial path...")
    wrapper = ProTokenWrapper(
        ckpt_path=args.ckpt,
        encoder_config_path=args.encoder_config,
        decoder_config_path=args.decoder_config,
        vq_config_path=args.vq_config,
        padding_len=args.padding_len
    )
    evaluator = PathEvaluator()

    # Encode start and end points
    start_emb, _, _ = wrapper.encode(args.start_pdb)
    end_emb, _, _ = wrapper.encode(args.end_pdb)
    
    # Get features for decoding and the final coordinates
    feature, _, _ = protoken_basic_generator(args.start_pdb, NUM_RES=wrapper.padding_len, crop_start_idx_preset=0)
    end_coords = wrapper.decode(end_emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], "/dev/null")

    # Create the initial linear path
    current_path = []
    for i in range(args.max_steps + 1):
        alpha = i / args.max_steps
        interpolated_emb = (1 - alpha) * start_emb + alpha * end_emb
        current_path.append(interpolated_emb)
    
    # Calculate the energy of the initial path
    current_energy = evaluate_path_energy(current_path, evaluator, end_coords, wrapper, feature)
    print(f"Initial path energy: {current_energy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Main Metropolis Loop with Simulated Annealing
    print(f"Starting Metropolis search with {args.num_iterations} iterations...")
    
    for i in tqdm(range(args.num_iterations)):
        # Annealing schedule for temperature T
        temperature = args.temp_initial * (1.0 - i / args.num_iterations)

        # Propose a new path by perturbing a random intermediate point
        new_path = list(current_path)
        idx_to_perturb = np.random.randint(1, args.max_steps) # Excludes start and end points
        
        perturbation = np.random.normal(loc=0.0, scale=args.noise_scale, size=new_path[idx_to_perturb].shape)
        new_path[idx_to_perturb] += perturbation

        # Calculate energy of the new path
        new_energy = evaluate_path_energy(new_path, evaluator, end_coords, wrapper, feature)

        # Metropolis acceptance criterion
        delta_energy = new_energy - current_energy
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            # Accept the new path
            current_path = new_path
            current_energy = new_energy

        if i % args.log_interval == 0:
            print(f"Iteration {i}: Current Energy = {current_energy:.4f}, Temp = {temperature:.4f}")

    # 3. Save the final optimized path
    print("Search finished. Saving final path...")
    for i, emb in enumerate(current_path):
        output_pdb_path = os.path.join(args.output_dir, f"metropolis_path_{i}.pdb")
        wrapper.decode(emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], output_pdb_path)

    print(f"Final path saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Metropolis path search for protein conformation transitions.')
    
    # Inputs
    parser.add_argument('--start_pdb', type=str, required=True)
    parser.add_argument('--end_pdb', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--encoder_config', type=str, required=True)
    parser.add_argument('--decoder_config', type=str, required=True)
    parser.add_argument('--vq_config', type=str, required=True)

    # Search Parameters
    parser.add_argument('--max_steps', type=int, default=30, help='Number of intermediate steps in the path.')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Total number of Metropolis iterations.')
    parser.add_argument('--noise_scale', type=float, default=0.05, help='Scale of the random noise for perturbations.')
    parser.add_argument('--temp_initial', type=float, default=1.0, help='Initial temperature for simulated annealing.')
    parser.add_argument('--padding_len', type=int, default=768)
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()
    metropolis_search(args)
