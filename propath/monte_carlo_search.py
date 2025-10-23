
import jax.numpy as jnp
import numpy as np
import argparse
import os
from tqdm import tqdm

import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
sys.path.append('..')

from PROTOKEN.data.dataset import protoken_basic_generator
# Import our custom modules
from propath.rl_path_finder import ProTokenWrapper, PathEvaluator

def monte_carlo_search(args):
    """Performs a greedy Monte Carlo search to find a path."""

    # 1. Initialize Environment components
    print("Initializing wrapper and evaluator...")
    wrapper = ProTokenWrapper(
        ckpt_path=args.ckpt,
        encoder_config_path=args.encoder_config,
        decoder_config_path=args.decoder_config,
        vq_config_path=args.vq_config,
        padding_len=args.padding_len
    )
    evaluator = PathEvaluator()

    # 2. Encode start and end structures
    print("Encoding start and end structures...")
    start_emb, _, _ = wrapper.encode(args.start_pdb)
    end_emb, _, _ = wrapper.encode(args.end_pdb)
    
    # Get features needed for decoding
    feature, _, _ = protoken_basic_generator(args.start_pdb, NUM_RES=wrapper.padding_len, crop_start_idx_preset=0)
    end_coords = wrapper.decode(end_emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], "/dev/null")

    # 3. Main Search Loop
    print(f"Starting Monte Carlo search with {args.max_steps} steps and {args.num_samples} samples per step...")
    os.makedirs(args.output_dir, exist_ok=True)

    current_emb = start_emb
    current_coords = wrapper.decode(current_emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], os.path.join(args.output_dir, "mc_path_0.pdb"))

    for step in tqdm(range(1, args.max_steps + 1)):
        best_next_emb = None
        best_next_coords = None
        best_reward = -np.inf

        # Generate and evaluate N random samples for the next step
        for _ in range(args.num_samples):
            # Generate a random perturbation
            # The scale of the noise is a critical hyperparameter
            random_delta = np.random.normal(loc=0.0, scale=args.noise_scale, size=current_emb.shape) # type: ignore
            
            # Add to current embedding to get a candidate next step
            candidate_emb = current_emb + random_delta
            candidate_coords = wrapper.decode(candidate_emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], "/dev/null")

            # Evaluate this step using the same reward logic as the RL env
            step_rmsd = evaluator.calculate_rmsd(current_coords, candidate_coords)
            smoothness_reward = -step_rmsd

            dist_to_target_prev = evaluator.calculate_rmsd(current_coords, end_coords)
            dist_to_target_current = evaluator.calculate_rmsd(candidate_coords, end_coords)
            progress_reward = dist_to_target_prev - dist_to_target_current

            total_reward = smoothness_reward + progress_reward

            if total_reward > best_reward:
                best_reward = total_reward
                best_next_emb = candidate_emb
                best_next_coords = candidate_coords

        # Take the best step found
        current_emb = best_next_emb
        current_coords = best_next_coords
        
        # Save the PDB for the best step
        wrapper.decode(current_emb, feature['seq_mask'], feature['residue_index'], feature['aatype'], os.path.join(args.output_dir, f"mc_path_{step}.pdb"))
        print(f"Step {step}: Best reward = {best_reward:.4f}")

    print("Monte Carlo search finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monte Carlo search for protein path finding.')
    
    # Structure inputs
    parser.add_argument('--start_pdb', type=str, required=True, help='Path to the starting PDB file.')
    parser.add_argument('--end_pdb', type=str, required=True, help='Path to the ending PDB file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated PDB path.')

    # Model inputs
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pkl file).')
    parser.add_argument('--encoder_config', type=str, required=True, help='Path to the encoder config yaml.')
    parser.add_argument('--decoder_config', type=str, required=True, help='Path to the decoder config yaml.')
    parser.add_argument('--vq_config', type=str, required=True, help='Path to the VQ config yaml.')

    # Search parameters
    parser.add_argument('--max_steps', type=int, default=30, help='Number of steps in the path.')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of random samples to evaluate at each step.')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='Scale of the random noise for perturbations.')
    parser.add_argument('--padding_len', type=int, default=768, help="Padding length for the model.")

    args = parser.parse_args()
    monte_carlo_search(args)
