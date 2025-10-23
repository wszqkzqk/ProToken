#!/usr/bin/env python3

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
sys.path.append('..')

import jax
import sys
import numpy as np
import argparse
import jax
import jax.numpy as jnp
import pickle as pkl
from functools import partial
from flax import linen as nn

# MDAnalysis for RMSD calculation
import MDAnalysis as mda
from MDAnalysis.analysis import rms

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ProToken specific imports
from PROTOKEN.model.encoder import VQ_Encoder
from PROTOKEN.model.decoder import VQ_Decoder, Protein_Decoder
from PROTOKEN.data.protein_utils import save_pdb_from_aux
from PROTOKEN.common.config_load import load_config
from PROTOKEN.data.dataset import protoken_basic_generator
from PROTOKEN.data.utils import make_2d_features
from PROTOKEN.config.global_config import GLOBAL_CONFIG

class ProTokenWrapper:
    """
    A wrapper class to handle ProToken model loading, encoding, and decoding.
    """
    def __init__(self, ckpt_path, encoder_config_path, decoder_config_path, vq_config_path, padding_len=768):
        print("Initializing ProTokenWrapper...")
        self.padding_len = padding_len
        self.exclude_neighbor = 3
        GLOBAL_CONFIG['use_dropout'] = False

        # Load configs
        self.encoder_cfg = load_config(encoder_config_path)
        self.decoder_cfg = load_config(decoder_config_path)
        self.vq_cfg = load_config(vq_config_path)
        self.encoder_cfg.seq_len = self.padding_len
        self.decoder_cfg.seq_len = self.padding_len

        # Initialize models
        self._init_models()

        # Load parameters
        self._load_params(ckpt_path)
        
        # Jit functions
        self._jit_functions()
        print("ProTokenWrapper initialized successfully.")

    def _init_models(self):
        """Initializes the encoder and decoder models."""
        self.encoder = VQ_Encoder(GLOBAL_CONFIG, self.encoder_cfg)
        self.vq_decoder = VQ_Decoder(GLOBAL_CONFIG, self.decoder_cfg, pre_layer_norm=False)
        self.protein_decoder = Protein_Decoder(GLOBAL_CONFIG, self.decoder_cfg)
        self.project_out = nn.Dense(features=self.vq_cfg.dim_in, kernel_init=nn.initializers.lecun_normal(), use_bias=False)

    def _load_params(self, ckpt_path):
        """Loads model parameters from a checkpoint file."""
        print(f"Loading checkpoint from {ckpt_path}...")
        with open(ckpt_path, "rb") as f:
            self.params = pkl.load(f)
        self.params = jax.tree_map(lambda x: jnp.array(x), self.params)

    def _jit_functions(self):
        """Creates JIT-compiled versions of encode and decode functions for performance."""
        
        # Encoder part
        def encode_fn(params, batch_input):
            # Note: We only need the encoder part of the inference cell
            # Directly calling the encoder
            return self.encoder.apply(params, *batch_input)

        # Decoder part
        def decode_fn(params, embedding, seq_mask, residue_index, fake_aatype):
            # This function mimics the decoding process from a continuous embedding
            vq_act_project_out = self.project_out.apply({'params': params['project_out']}, embedding)
            
            single_act_decode, pair_act_decode, _, _ = self.vq_decoder.apply(
                {'params': params['vq_decoder']}, vq_act_project_out, seq_mask, residue_index
            )
            
            final_atom_positions, _, _, _, _, _ = self.protein_decoder.apply(
                {'params': params['protein_decoder']}, single_act_decode, pair_act_decode, seq_mask, fake_aatype
            )
            return final_atom_positions

        # We need to get the correct parameter sub-trees
        encoder_params = {'params': self.params['params']['vq_encoder']}
        decoder_params = {
            'project_out': self.params['params']['project_out'],
            'vq_decoder': self.params['params']['vq_decoder'],
            'protein_decoder': self.params['params']['protein_decoder']
        }

        # Jit the functions
        self.jitted_encode = jax.jit(partial(encode_fn, encoder_params))
        self.jitted_decode = jax.jit(decode_fn, static_argnums=(2,3,4))


    def encode(self, pdb_path):
        """Encodes a PDB file into a continuous latent embedding."""
        feature, _, seq_len = protoken_basic_generator(pdb_path, NUM_RES=self.padding_len, crop_start_idx_preset=0)
        batch_feature = jax.tree_map(lambda x: jnp.array(x)[None, ...], feature) # Add batch dimension
        batch_feature = make_2d_features(batch_feature, self.padding_len, self.exclude_neighbor)
        
        protoken_feature_input = ["seq_mask", "aatype", "fake_aatype", "residue_index",
                                  "template_all_atom_masks", "template_all_atom_positions", "template_pseudo_beta",
                                  "backbone_affine_tensor", "backbone_affine_tensor_label", 
                                  "torsion_angles_sin_cos", "torsion_angles_mask", "atom14_atom_exists",
                                  "dist_gt_perms", "dist_mask_perms", "perms_padding_mask"]

        batch_input = [batch_feature[name] for name in protoken_feature_input]
        
        # The VQ_Encoder returns multiple values, we are interested in the pre-quantized embedding
        # Based on VQ_Encoder structure, the first output is the embedding
        embedding, _ = self.jitted_encode(batch_input)
        
        return embedding[0], feature['seq_mask'], seq_len # Remove batch dimension

    def decode(self, embedding, seq_mask, residue_index, aatype, output_pdb_path):
        """Decodes a continuous latent embedding into a PDB file."""
        
        # Add batch dimension for model
        embedding = embedding[None, ...]
        seq_mask = seq_mask[None, ...]
        residue_index = residue_index[None, ...]
        fake_aatype = jnp.ones_like(seq_mask, dtype=jnp.int32) * 7 # Glycine

        final_atom_positions = self.jitted_decode(self.params['params'], embedding, seq_mask, residue_index, fake_aatype)
        
        # Remove batch dimension
        final_atom_positions = final_atom_positions[0]
        seq_mask = seq_mask[0]
        residue_index = residue_index[0]
        
        # Prepare for saving
        atom_mask = np.array([1,1,1,0,1]+[0]*32, dtype=np.float32) * seq_mask[..., None]
        
        aux_result = {
            "aatype": aatype,
            "residue_index": residue_index,
            "atom_positions": np.array(final_atom_positions),
            "atom_mask": atom_mask,
            "plddt": np.ones_like(seq_mask) * 100.0
        }
        
        save_pdb_from_aux(aux_result, output_pdb_path)
        print(f"Saved decoded structure to {output_pdb_path}")
        return aux_result['atom_positions']


class PathEvaluator:
    """
    A class to evaluate the quality of a generated protein path.
    """
    def calculate_rmsd(self, coords1, coords2, selection="name CA"):
        """Calculates RMSD between two sets of coordinates in memory."""
        
        # Create Universe objects from coordinates
        # We need number of residues and atoms to create the universe properly
        n_residues = coords1.shape[0]
        
        # Create a dummy universe, we will overwrite coordinates
        u1 = mda.Universe.empty(n_residues=n_residues, n_atoms=coords1.shape[1], atom_resindex=np.arange(n_residues), trajectory=True)
        u2 = mda.Universe.empty(n_residues=n_residues, n_atoms=coords2.shape[1], atom_resindex=np.arange(n_residues), trajectory=True)
        
        u1.atoms.positions = coords1
        u2.atoms.positions = coords2
        
        # For selection to work, we need atom names. Let's assume a simple topology.
        # This part is tricky and might need adjustment based on ProToken's output atom order.
        # For now, let's assume the first 4 atoms are N, CA, C, O for simplicity with CA-RMSD.
        u1.add_TopologyAttr('name', ['N', 'CA', 'C', 'O'] + ['X'] * (coords1.shape[1] - 4))
        u2.add_TopologyAttr('name', ['N', 'CA', 'C', 'O'] + ['X'] * (coords2.shape[1] - 4))

        return rms.rmsd(u1.select_atoms(selection).positions, u2.select_atoms(selection).positions, superposition=True)


class ProteinPathEnv:
    """
    A reinforcement learning environment for finding protein conformation transition paths.
    Implements a 'guided exploration' strategy where the agent learns a correction to a linear baseline path.
    """
    def __init__(self, wrapper: ProTokenWrapper, evaluator: PathEvaluator, start_pdb_path: str, end_pdb_path: str, max_steps=30):
        self.wrapper = wrapper
        self.evaluator = evaluator
        self.max_steps = max_steps
        self.step_size = 1.0 / max_steps # The fraction of the total path to traverse in one linear step

        # Encode start and end structures
        print("Initializing RL Environment: Encoding start and end structures...")
        self.start_emb, self.start_mask, self.start_len = self.wrapper.encode(start_pdb_path)
        self.end_emb, self.end_mask, self.end_len = self.wrapper.encode(end_pdb_path)
        
        # Decode them once to get the ground truth coordinates for reward calculation
        self.feature, _, _ = protoken_basic_generator(start_pdb_path, NUM_RES=self.wrapper.padding_len, crop_start_idx_preset=0)
        self.start_coords = self.wrapper.decode(self.start_emb, self.feature['seq_mask'], self.feature['residue_index'], self.feature['aatype'], "/dev/null")
        self.end_coords = self.wrapper.decode(self.end_emb, self.feature['seq_mask'], self.feature['residue_index'], self.feature['aatype'], "/dev/null")

        self.action_space = None
        self.observation_space = None

    def reset(self):
        """Resets the environment to the starting state."""
        self.current_step = 0
        self.current_emb = self.start_emb
        self.current_coords = self.start_coords
        
        state = (self.current_emb, self.end_emb)
        return state

    def step(self, delta_action):
        """
        Execute one time step using guided exploration.
        The agent's action is a correction (delta) to the linear interpolation baseline.
        """
        self.current_step += 1

        # 1. Calculate the baseline linear step
        linear_step_vector = (self.end_emb - self.current_emb) * self.step_size

        # 2. Apply the agent's correction (delta_action) to the baseline
        final_step_vector = linear_step_vector + delta_action
        next_emb = self.current_emb + final_step_vector

        # 3. Decode the new embedding to get 3D structure
        next_coords = self.wrapper.decode(next_emb, self.feature['seq_mask'], self.feature['residue_index'], self.feature['aatype'], "/dev/null")

        # 4. Calculate reward
        reward = self._calculate_reward(self.current_coords, next_coords)

        # 5. Update state
        self.current_emb = next_emb
        self.current_coords = next_coords
        
        # 6. Check for termination
        done = self.current_step >= self.max_steps
        
        next_state = (self.current_emb, self.end_emb)
        info = {}

        return next_state, reward, done, info

    def _calculate_reward(self, prev_coords, current_coords):
        """
        Calculates the reward for a transition.
        A simple reward: negative RMSD to encourage smoothness and progress.
        """
        # Reward for smoothness (low RMSD between steps)
        step_rmsd = self.evaluator.calculate_rmsd(prev_coords, current_coords)
        smoothness_reward = -step_rmsd

        # Reward for getting closer to the target
        dist_to_target_prev = self.evaluator.calculate_rmsd(prev_coords, self.end_coords)
        dist_to_target_current = self.evaluator.calculate_rmsd(current_coords, self.end_coords)
        progress_reward = dist_to_target_prev - dist_to_target_current

        # Combine rewards
        total_reward = smoothness_reward + progress_reward
        return total_reward

def main(args):
    # Define paths
    start_pdb_path = args.start_pdb
    end_pdb_path = args.end_pdb
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the wrapper
    wrapper = ProTokenWrapper(
        ckpt_path=args.ckpt,
        encoder_config_path=args.encoder_config,
        decoder_config_path=args.decoder_config,
        vq_config_path=args.vq_config,
        padding_len=args.padding_len
    )

    # This part demonstrates the original linear interpolation workflow
    if args.workflow == 'interpolation':
        print("Running linear interpolation workflow...")
        # 1. Encode start and end structures
        print("Encoding start and end structures...")
        start_emb, _, start_len = wrapper.encode(start_pdb_path)
        end_emb, _, end_len = wrapper.encode(end_pdb_path)
        
        if start_len != end_len:
            print(f"Warning: Start and end PDBs have different lengths ({start_len} vs {end_len}). This might lead to issues.")

        # 2. Linear Interpolation
        print(f"Performing linear interpolation with {args.steps} steps...")
        feature, _, _ = protoken_basic_generator(start_pdb_path, NUM_RES=wrapper.padding_len, crop_start_idx_preset=0)

        for i in range(args.steps + 1):
            alpha = i / args.steps
            interpolated_emb = (1 - alpha) * start_emb + alpha * end_emb
            
            output_pdb_path = os.path.join(output_dir, f"interpolated_{i}.pdb")
            
            # 3. Decode interpolated embedding
            wrapper.decode(
                interpolated_emb, 
                feature['seq_mask'], 
                feature['residue_index'],
                feature['aatype'],
                output_pdb_path
            )
        print("Interpolation workflow finished. Interpolated PDB files are in:", output_dir)

    # This part demonstrates a dummy run of the RL environment
    elif args.workflow == 'rl_test':
        print("Running RL environment test workflow...")
        evaluator = PathEvaluator()
        env = ProteinPathEnv(wrapper, evaluator, start_pdb_path, end_pdb_path)
        
        state = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            # In a real scenario, an agent would produce this action
            # Here, we use a dummy action: a small step towards the target embedding
            action = (env.end_emb - env.current_emb) * 0.1 
            
            next_state, reward, done, _ = env.step(action)
            
            print(f"Step {step + 1}: Reward = {reward:.4f}")
            total_reward += reward
            
            if done:
                print("Episode finished.")
                break
        
        print(f"Total reward for dummy episode: {total_reward:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated workflow for protein path generation.')
    
    # Structure inputs
    parser.add_argument('--start_pdb', type=str, required=True, help='Path to the starting PDB file.')
    parser.add_argument('--end_pdb', type=str, required=True, help='Path to the ending PDB file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save interpolated PDB files.')

    # Model inputs
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pkl file).')
    parser.add_argument('--encoder_config', type=str, required=True, help='Path to the encoder config yaml.')
    parser.add_argument('--decoder_config', type=str, required=True, help='Path to the decoder config yaml.')
    parser.add_argument('--vq_config', type=str, required=True, help='Path to the VQ config yaml.')

    # Workflow selection
    parser.add_argument('--workflow', type=str, default='interpolation', choices=['interpolation', 'rl_test'], help='The workflow to run.')

    # Workflow parameters
    parser.add_argument('--steps', type=int, default=20, help='Number of interpolation steps.')
    parser.add_argument('--padding_len', type=int, default=768, help="Padding length for the model.")

    args = parser.parse_args()
    
    main(args)
