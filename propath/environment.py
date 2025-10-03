#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from functools import partial
from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController

from PROTOKEN.model.decoder import VQ_Decoder, Protein_Decoder

class PathDecodingEnvironment:
    """A black-box environment to decode a latent path into 3D coordinates."""

    def __init__(self, models):
        """
        Initializes the environment with all the necessary pre-loaded models.
        Args:
            models: A dictionary containing all model components and parameters.
        """
        self.models = models
        self.jit_solve_ode = partial(jax.jit, static_argnums=(0, 1, 2))(self._solve_ode)
        self.jit_run_structure_decoder = jax.jit(self._run_structure_decoder)

    def _solve_ode(self, t_0, t_1, dt0, x_in, seq_mask, residue_index):
        """Core JIT-compiled ODE solver."""
        dit_model = self.models["dit"]["model"]
        dit_params = self.models["dit"]["params"]
        scheduler = self.models["dit"]["scheduler"]
        
        indicator = dit_params['params']['protoken_indicator']
        indicator = jnp.concatenate([indicator, dit_params['params']['aatype_indicator']], axis=-1)
        
        def ode_drift(t, y, args):
            t_arr = jnp.full((y.shape[0],), t)
            eps_prime = dit_model.apply({'params': dit_params['params']['model']}, y + indicator[None, ...], 
                                    seq_mask, t_arr, tokens_rope_index=residue_index)
            beta_t = scheduler.betas[jnp.int32(t)]
            sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[jnp.int32(t)]
            return 0.5 * beta_t * (-y + 1.0 / sqrt_one_minus_alphas_cumprod_t * eps_prime)

        term = ODETerm(ode_drift)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        sol = diffeqsolve(term, solver, t0=t_0, t1=t_1, y0=x_in, dt0=dt0, stepsize_controller=stepsize_controller)
        return sol.ys[-1]

    def _run_structure_decoder(self, vq_decoder_input, seq_mask, residue_index, aatype_tokens):
        """Core JIT-compiled structure decoder."""
        distiller = self.models['protoken_distiller']
        params = distiller['params']

        single_act_decode, pair_act_decode, _, _ = distiller['vq_decoder'].apply(
            params['vq_decoder'], vq_decoder_input, seq_mask, residue_index
        )

        final_atom_positions, _, _, _, _, _ = distiller['protein_decoder'].apply(
            params['protein_decoder'], single_act_decode, pair_act_decode, seq_mask, aatype_tokens
        )
        
        return final_atom_positions

    def decode_path_to_coords(self, xT_path, seq_mask, residue_index, aatype_tokens):
        """
        Decodes a batch of latent vectors (a path) into a batch of 3D coordinates (a trajectory).

        Args:
            xT_path: Latent vectors representing the path. Shape: (n_points, n_res, latent_dim)
            seq_mask: The sequence mask for the protein. Shape: (n_res,)
            residue_index: The residue index for the protein. Shape: (n_res,)
            aatype_tokens: The amino acid sequence tokens. Shape: (n_res,)

        Returns:
            coords_path: The decoded 3D coordinates trajectory. Shape: (n_points, n_res, 37, 3)
        """
        print(f"Decoding path of {xT_path.shape[0]} points...")
        n_points, n_res, _ = xT_path.shape

        # Ensure inputs have a batch dimension for the solver
        seq_mask_batch = jnp.repeat(seq_mask[None, ...], n_points, axis=0)
        residue_index_batch = jnp.repeat(residue_index[None, ...], n_points, axis=0)
        aatype_tokens_batch = jnp.repeat(aatype_tokens[None, ...], n_points, axis=0)

        # --- Step 1: Reverse ODE (xT -> x0) ---
        print("  - Step 1: Running reverse ODE...")
        scheduler = self.models["dit"]["scheduler"]
        x0_path = self.jit_solve_ode(scheduler.num_timesteps, 0, -1.0, xT_path, seq_mask_batch, residue_index_batch)

        # --- Step 2: Hard Token Lookup (x0 -> indices) ---
        print("  - Step 2: Performing hard token lookup...")
        protoken_emb = self.models['protoken_emb']
        x0_protoken_part = x0_path[..., :protoken_emb.shape[-1]]
        
        # Calculate distances and find the argmin (hard lookup)
        distances = jnp.sum((x0_protoken_part[:, :, None, :] - protoken_emb[None, None, :, :])**2, axis=-1)
        protoken_indices = jnp.argmin(distances, axis=-1)

        # --- Step 3: Structure Decoder (indices -> coords) ---
        print("  - Step 3: Running structure decoder...")
        distiller = self.models['protoken_distiller']
        params = distiller['params']
        vq_codebook = params['vq_tokenizer']['params']['codebook']

        # Look up the VQ codebook and apply the projection layer
        vq_act = vq_codebook[protoken_indices]
        vq_decoder_input = distiller['project_out'].apply(params['project_out'], vq_act)

        # Run the final decoder models
        coords_path = self.jit_run_structure_decoder(vq_decoder_input, seq_mask_batch, residue_index_batch, aatype_tokens_batch)

        print("Path decoding complete.")
        return coords_path