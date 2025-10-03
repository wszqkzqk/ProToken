#!/usr/bin/env python3

import argparse
import os
import sys

# Ensure the project root is in the Python path
# This allows us to import from src, PROTOKEN, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
from functools import partial

# import optax # To be used for optimization

# --- Model and Config Imports ---
from flax import linen as nn
from jax.tree_util import tree_map

# Stage 1 (DiT) imports
from src.model.diffusion_transformer import DiffusionTransformer
from train.schedulers import GaussianDiffusion
from configs.dit_config import dit_config
from configs.global_config import global_config as dit_global_config

# Stage 2 (Decoder) imports
from PROTOKEN.model.decoder import VQ_Decoder, Protein_Decoder
from PROTOKEN.common.config_load import load_config as load_decoder_config
from PROTOKEN.config.global_config import GLOBAL_CONFIG as decoder_global_config


def parse_arguments():
    """Parses command-line arguments for the optimization workflow."""
    parser = argparse.ArgumentParser(
        description="Run gradient-based optimization in ProToken's latent space to find protein conformational transition paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output Arguments ---
    parser.add_argument("--pdb_start", type=str, required=True,
                        help="Path to the starting PDB file.")
    parser.add_argument("--pdb_end", type=str, required=True,
                        help="Path to the ending PDB file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the optimized trajectory, logs, and other results.")

    # --- Optimization Arguments ---
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of optimization steps.")

    # --- Path Representation Arguments ---
    parser.add_argument("--n_points", type=int, default=100,
                        help="Number of intermediate points to represent the transition path.")
    parser.add_argument("--path_type", type=str, default="linear", choices=["linear", "bezier"],
                        help="Type of path parameterization in the latent space.")

    # --- Model & Environment Arguments ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args

def load_models(project_root):
    """
    Loads all necessary models, parameters, and configs for the end-to-end workflow.
    """
    print("Loading models...")

    # --- Define Paths ---
    paths = {
        "dit_params": os.path.join(project_root, "ckpts/PT_DiT_params_1000000.pkl"),
        "decoder_params": os.path.join(project_root, "ckpts/protoken_params_100000.pkl"),
        "decoder_config": os.path.join(project_root, "PROTOKEN/config/decoder.yaml"),
        "vq_config": os.path.join(project_root, "PROTOKEN/config/vq.yaml"),
        "encoder_config": os.path.join(project_root, "PROTOKEN/config/encoder.yaml"), # Added for VQ_Encoder
        "protoken_emb": os.path.join(project_root, "embeddings/protoken_emb.pkl"),
        "aatype_emb": os.path.join(project_root, "embeddings/aatype_emb.pkl"),
    }

    # --- Load Embeddings ---
    with open(paths["protoken_emb"], 'rb') as f:
        protoken_emb = jnp.array(pkl.load(f), dtype=jnp.float32)
    with open(paths["aatype_emb"], 'rb') as f:
        aatype_emb = jnp.array(pkl.load(f), dtype=jnp.float32)

    # --- Load Stage 1: DiT Model ---
    dit_global_config.dropout_flag = False
    dit_model = DiffusionTransformer(config=dit_config, global_config=dit_global_config)
    with open(paths["dit_params"], "rb") as f:
        dit_params = pkl.load(f)
        dit_params = tree_map(lambda x: jnp.array(x), dit_params)
    
    scheduler = GaussianDiffusion(num_diffusion_timesteps=500)

    # --- Load Stage 2: VQ-VAE Components (Encoder + Decoder) ---
    encoder_cfg = load_decoder_config(paths["encoder_config"])
    decoder_cfg = load_decoder_config(paths["decoder_config"])
    vq_cfg = load_decoder_config(paths["vq_config"])
    encoder_cfg.seq_len = 512 # Default padding length
    decoder_cfg.seq_len = 512
    
    decoder_global_config['use_dropout'] = False
    
    # Initialize models
    vq_encoder_model = VQ_Encoder(decoder_global_config, encoder_cfg)
    vq_tokenizer_model = VQTokenizer(vq_cfg, dtype=jnp.float32)
    vq_decoder_model = VQ_Decoder(global_config=decoder_global_config, cfg=decoder_cfg, pre_layer_norm=False)
    protein_decoder_model = Protein_Decoder(global_config=decoder_global_config, cfg=decoder_cfg)
    project_in_model = nn.Dense(features=vq_cfg.dim_code + 20, kernel_init=nn.initializers.lecun_normal(), use_bias=False)
    project_out_model = nn.Dense(features=vq_cfg.dim_in, kernel_init=nn.initializers.lecun_normal(), use_bias=False)

    # Load all params from the single checkpoint
    with open(paths["decoder_params"], "rb") as f:
        params_raw = pkl.load(f)["params"]
        params_raw = tree_map(lambda x: jnp.array(x), params_raw)

    # Structure params correctly for each model component
    protoken_distiller_params = {
        "vq_encoder": {"params": params_raw["encoder"]},
        "vq_tokenizer": {"params": params_raw["vq_tokenizer"]},
        "vq_decoder": {"params": params_raw["vq_decoder"]},
        "protein_decoder": {"params": params_raw["protein_decoder"]},
        "project_in": {"params": params_raw["project_in"]},
        "project_out": {"params": params_raw["project_out"]},
    }

    models = {
        "protoken_emb": protoken_emb,
        "aatype_emb": aatype_emb,
        "dit": {
            "model": dit_model,
            "params": dit_params,
            "scheduler": scheduler,
        },
        "protoken_distiller": {
            "vq_encoder": vq_encoder_model,
            "vq_tokenizer": vq_tokenizer_model,
            "vq_decoder": vq_decoder_model,
            "protein_decoder": protein_decoder_model,
            "project_in": project_in_model,
            "project_out": project_out_model,
            "params": protoken_distiller_params,
        }
    }
    
    print("Models loaded successfully.")
    return models

def encode_structure(pdb_path, models, args):
    """
    Encodes a PDB structure into the DiT latent space.
    This involves PDB parsing, tokenization, embedding lookup, and running the forward ODE.
    """
    print(f"Encoding {os.path.basename(pdb_path)}...")

    # --- 1. PDB Parsing ---
    NRES = 512 # TODO: Move to a config object
    EXCLUDE_NEIGHBOR = 3
    feature, _, seq_len = protoken_basic_generator(pdb_path, NUM_RES=NRES, crop_start_idx_preset=0)
    batch_feature = tree_map(lambda x: jnp.array(x)[None, ...], feature) # Add batch dimension
    batch_feature = make_2d_features(batch_feature, NRES, EXCLUDE_NEIGHBOR)

    # --- 2. VQ-VAE Encoder Forward Pass (PDB -> Tokens) ---
    distiller = models["protoken_distiller"]
    distiller_params = distiller["params"]
    
    # Prepare inputs for the encoder
    protoken_feature_input_names = [
        "seq_mask", "aatype", "fake_aatype", "residue_index",
        "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
        "backbone_affine_tensor", "backbone_affine_tensor_label", 
        "torsion_angles_sin_cos", "torsion_angles_mask", "atom14_atom_exists",
        "dist_gt_perms", "dist_mask_perms", "perms_padding_mask"
    ]
    batch_input = [batch_feature[name] for name in protoken_feature_input_names]
    true_aatype = batch_feature['aatype']

    # Call VQ_Encoder
    single_act, _, _ = distiller["vq_encoder"].apply(distiller_params["vq_encoder"], *batch_input)

    # Call Projector
    single_act_project_in = distiller["project_in"].apply(distiller_params["project_in"], single_act)

    # Call VQ_Tokenizer
    vq_dim = distiller["vq_tokenizer"].vq_cfg.dim_code
    single_act_for_vq = single_act_project_in[..., :vq_dim]
    _, quantize_results = distiller["vq_tokenizer"].apply(distiller_params["vq_tokenizer"], single_act_for_vq, batch_feature['seq_mask'])
    
    protokens = quantize_results["encoding_indices"].squeeze(0) # Remove batch dim
    aatype_tokens = true_aatype.squeeze(0) # Remove batch dim

    # --- 3. Generate Initial Embedding (x0) ---
    protoken_emb = models["protoken_emb"]
    aatype_emb = models["aatype_emb"]
    
    embedding = np.concatenate(
        [protoken_emb[protokens.astype(np.int32)],
         aatype_emb[aatype_tokens.astype(np.int32)]], axis=-1)
    embedding = np.pad(embedding, ((0, NRES - seq_len), (0, 0)))
    x0 = jnp.array(embedding)[None, ...] # Add batch dimension

    # --- 4. Run Forward ODE (x0 -> xT) ---
    seq_mask = batch_feature['seq_mask']
    residue_index = batch_feature['residue_index']
    ndevices = len(jax.devices())
    
    # Reshape for pmap if necessary, for now assume single device
    if x0.shape[0] % ndevices != 0:
        # Simple padding if batch size doesn't fit devices. A real implementation should handle this better.
        # For single PDB encoding, batch size is 1, so this is important for multi-device setups.
        # Here, we'll just reshape for a single device to keep it simple.
        pass # Assuming single device execution for now.

    dit_model = models["dit"]["model"]
    dit_params = models["dit"]["params"]
    scheduler = models["dit"]["scheduler"]

    # Define and jit the ODE solver function
    @partial(jax.jit, static_argnums=(0,1,2))
    def solve_ode(t_0, t_1, dt0, x_0_inp, seq_mask_inp, residue_index_inp):
        from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController
        indicator = dit_params['params']['protoken_indicator']
        indicator = jnp.concatenate([indicator, dit_params['params']['aatype_indicator']], axis=-1)
        
        def ode_drift(t, y, args):
            t_arr = jnp.full((y.shape[0],), t)
            eps_prime = dit_model.apply({'params': dit_params['params']['model']}, y + indicator[None, ...], 
                                    seq_mask_inp, t_arr, tokens_rope_index=residue_index_inp)
            beta_t = scheduler.betas[jnp.int32(t)]
            sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[jnp.int32(t)]
            return 0.5 * beta_t * (-y + 1.0 / sqrt_one_minus_alphas_cumprod_t * eps_prime)

        term = ODETerm(ode_drift)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        sol = diffeqsolve(term, solver, t0=t_0, t1=t_1, y0=x_0_inp, dt0=dt0, stepsize_controller=stepsize_controller)
        return sol.ys[-1]

    print("Running forward ODE to get latent representation...")
    xT = solve_ode(0, scheduler.num_timesteps, 1.0, x0, seq_mask, residue_index)
    
    latent_representation = xT.squeeze(0) # Remove batch dim

    print("Encoding complete.")
    return latent_representation

def initialize_path(latent_start, latent_end, n_points, path_type="linear"):
    """
    Initializes the transition path in the latent space.
    For linear, it's a simple interpolation.
    For Bezier, it would involve initializing control points.
    """
    print(f"Initializing a {path_type} path with {n_points} points...")
    # TODO: Implement path initialization logic
    # For now, returning a simple linear interpolation
    lambdas = jnp.linspace(0, 1, n_points)[:, None, None]
    path = (1 - lambdas) * latent_start[None, ...] + lambdas * latent_end[None, ...]
    # For an optimizable path, we would return the parameters (e.g., control points)
    return path

def objective_function(path_coords, key):
    """
    Placeholder for the differentiable objective function.
    It takes the 3D coordinates of the structures along the path and returns a scalar loss.
    """
    # TODO: Implement objective function (e.g., RMSD smoothness, clash score)
    loss = 0.0
    return loss

def run_optimization_loop(args, models, initial_path):
    """
The main optimization loop."""
    print("Starting optimization...")
    # TODO: Set up optimizer (e.g., optax.adam)
    # TODO: Implement the main optimization loop
    # grad_fn = jax.value_and_grad(objective_function)
    
    # for step in range(args.n_steps):
    #     # 1. Decode path parameters to 3D coordinates (end-to-end)
    #     #    path_coords = decode_end_to_end(current_path_params, models)
    #     # 2. Calculate loss and gradients
    #     #    loss, grads = grad_fn(path_coords, key)
    #     # 3. Update path parameters
    #     #    updates, opt_state = optimizer.update(grads, opt_state)
    #     #    current_path_params = optax.apply_updates(current_path_params, updates)
    #     pass

    print("Optimization finished.")
    final_path = initial_path # Placeholder
    return final_path

def save_trajectory(path_coords, output_dir):
    """
    Saves the final path as a multi-model PDB file.
    """
    print(f"Saving final trajectory to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    # TODO: Implement PDB saving logic, similar to merge-pdbs.py
    print("Trajectory saved.")

def main():
    """Main function to orchestrate the workflow."""
    args = parse_arguments()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 1. Load Models
    models = load_models(project_root)

    # 2. Encode Start and End Structures
    latent_start = encode_structure(args.pdb_start, models)
    latent_end = encode_structure(args.pdb_end, models)

    # 3. Initialize Path
    # Note: For a real optimization, we would optimize the path *parameters*,
    # not the path points directly. This is a simplified representation.
    initial_path_params = initialize_path(latent_start, latent_end, args.n_points, args.path_type)

    # 4. Run Optimization
    final_path_params = run_optimization_loop(args, models, initial_path_params)

    # 5. Decode final path and save
    # final_coords = decode_end_to_end(final_path_params, models) # Placeholder
    # save_trajectory(final_coords, args.output_dir)

    print("\nWorkflow scaffolding complete. Next steps are to implement the placeholder functions.")

if __name__ == "__main__":
    main()
