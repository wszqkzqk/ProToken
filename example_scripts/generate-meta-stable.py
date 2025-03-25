#!/usr/bin/env python3

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import pickle as pkl
import numpy as np
from tqdm import tqdm
import argparse
import math

from functools import partial
from src.model.diffusion_transformer import DiffusionTransformer
from train.schedulers import GaussianDiffusion
import datetime
from flax.jax_utils import replicate
from functools import reduce

from configs.global_config import global_config
from configs.dit_config import dit_config
global_config.dropout_flag = False 

# Add configuration class to manage model parameters
class ModelConfig:
    def __init__(self, batch_size=100, nres=512):
        # Basic settings
        self.nres = nres
        self.dim_emb = None  # Will be set after loading embeddings
        
        # Device and batch size
        self.ndevices = len(jax.devices())
        self.batch_size = batch_size
        self.adjust_batch_size()
        
    def adjust_batch_size(self):
        """Adjust batch_size to fit the number of devices"""
        self.nsample_per_device = math.ceil(self.batch_size / self.ndevices)
        self.batch_size = self.nsample_per_device * self.ndevices
        return self.batch_size
        
    def set_embedding_dim(self, protoken_dim, aatype_dim):
        """Set embedding dimension"""
        self.dim_emb = protoken_dim + aatype_dim
        
# Add interpolation function
def interpolate_latent(start_point, end_point, config, method='linear', 
                      distribution='uniform', num_points=None):
    # Determine the number of interpolation points
    num_points = num_points if num_points is not None else config.batch_size
    
    # Generate interpolation coefficients based on distribution type
    if distribution == 'uniform':
        lambda_arr = np.linspace(0, 1, num_points)
    elif distribution == 'gaussian':
        # Gaussian distribution, denser in the middle region
        x = np.linspace(-2, 2, num_points)
        lambda_arr = 1 / (1 + np.exp(-x))  # Sigmoid function maps values to (0,1)
        lambda_arr = (lambda_arr - lambda_arr[0]) / (lambda_arr[-1] - lambda_arr[0])  # Normalize to [0,1]
    elif distribution == 'logspace':
        # Logarithmic distribution, denser near the start
        lambda_arr = np.logspace(-3, 0, num_points)
        lambda_arr = lambda_arr / lambda_arr[-1]  # Normalize to [0,1]
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    
    # Calculate interpolation points based on method
    interpolated_points = []
    if method == 'linear':
        for i in range(num_points):
            interpolated_points.append(
                ((1.0 - lambda_arr[i]) * start_point + lambda_arr[i] * end_point)
            )
    elif method == 'spherical':
        # Spherical interpolation (SLERP)
        # Smooth rotational interpolation for vector data
        for i in range(num_points):
            t = lambda_arr[i]
            interp_point = (jnp.sin((1-t)*math.pi/2) * start_point + 
                           jnp.sin(t*math.pi/2) * end_point) / jnp.sin(math.pi/2)
            interpolated_points.append(interp_point)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
    # Convert to correct shape
    points_array = jnp.array(interpolated_points)
    
    # Ensure the number of points matches batch_size
    if num_points != config.batch_size:
        if num_points < config.batch_size:
            # Resample to more points
            indices = np.linspace(0, num_points-1, config.batch_size).astype(int)
            points_array = points_array[indices]
        else:
            # Truncate to batch_size
            points_array = points_array[:config.batch_size]
    
    # Reshape to device-friendly shape
    return points_array.reshape(config.ndevices, config.nsample_per_device, *points_array.shape[1:])

### Load embedding 
with open('../embeddings/protoken_emb.pkl', 'rb') as f:
    protoken_emb = jnp.array(pkl.load(f), dtype=jnp.float32)
with open('../embeddings/aatype_emb.pkl', 'rb') as f:
    aatype_emb = jnp.array(pkl.load(f), dtype=jnp.float32)

#### Initialize configuration
config = ModelConfig(batch_size=100, nres=512)
config.set_embedding_dim(protoken_emb.shape[-1], aatype_emb.shape[-1])

#### constants
NRES = config.nres
DIM_EMB = config.dim_emb
NDEVICES = config.ndevices
BATCH_SIZE = config.batch_size
NSAMPLE_PER_DEVICE = config.nsample_per_device

#### function utils 

def split_multiple_rng_keys(rng_key, num_keys):
    rng_keys = jax.random.split(rng_key, num_keys + 1)
    return rng_keys[:-1], rng_keys[-1]

def flatten_list_of_dicts(list_of_dicts):
    ### [{a: [1,2,3,4]}] -> [{a:1}, {a:2}, {a:3}, {a:4}]
    flattened_lists = [[{k: v[i] for k, v in d.items()} 
                        for i in range(len(next(iter(d.values()))))] for d in list_of_dicts]
    return reduce(lambda x, y: x+y, flattened_lists, [])

def protoken_emb_distance_fn(x, y):
    x_ = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    y_ = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-6)
    
    return -jnp.sum(x_ * y_, axis=-1)

def aatype_emb_distance_fn(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)

def aatype_index_to_resname(aatype_index):
    restypes = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
        'S', 'T', 'W', 'Y', 'V'
    ]
    
    return "".join([restypes[int(i)] for i in aatype_index])

def resname_to_aatype_index(resnames):
    restypes = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
        'S', 'T', 'W', 'Y', 'V'
    ]
    return np.array([restypes.index(a) for a in resnames], dtype=np.int32)

#### load model & params 
dit_model = DiffusionTransformer(
    config=dit_config, global_config=global_config
)
num_diffusion_timesteps = 500
scheduler = GaussianDiffusion(num_diffusion_timesteps=num_diffusion_timesteps)

#### rng keys
rng_key = jax.random.PRNGKey(8888)
np.random.seed(7777)

##### load params
ckpt_path = '../ckpts/PT_DiT_params_1000000.pkl'
with open(ckpt_path, "rb") as f:
    params = pkl.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

##### main inference functions
jit_apply_fn = jax.jit(dit_model.apply)
infer_protuple = True

def index_from_embedding(x):
    # x: (B, Nres, Nemb)
    protoken_indexes = \
        jnp.argmin(protoken_emb_distance_fn(x[..., None, :protoken_emb.shape[-1]], 
                                            protoken_emb[None, None, ...]), axis=-1)
    ret = {'protoken_indexes': protoken_indexes}
    if bool(infer_protuple):
        aatype_indexes = \
            jnp.argmin(aatype_emb_distance_fn(x[..., None, protoken_emb.shape[-1]:], 
                                                aatype_emb[None, None, ...]), axis=-1)
        ret.update({'aatype_indexes': aatype_indexes})
        
    return ret            
    
pjit_index_from_embedding = jax.pmap(jax.jit(index_from_embedding), axis_name="i")

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

def ode_drift(x, t, seq_mask, residue_index):
    t_arr = jnp.full((x.shape[0],), t)
    indicator = params['params']['protoken_indicator']
    if bool(infer_protuple):
        indicator = jnp.concatenate([indicator, params['params']['aatype_indicator']], 
                                    axis=-1)
    eps_prime = jit_apply_fn({'params': params['params']['model']}, x + indicator[None, ...], 
                            seq_mask, t_arr, tokens_rope_index=residue_index)

    beta_t = scheduler.betas[jnp.int32(t)]
    sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[jnp.int32(t)]
    
    return 0.5 * beta_t * (-x + 1.0 / sqrt_one_minus_alphas_cumprod_t * eps_prime)

rtol, atol, method = 1e-5, 1e-5, "RK45"

def solve_ode(t_0, t_1, dt0, x_0, seq_mask, residue_index):
    term = ODETerm(lambda t, y, args: jax.jit(ode_drift)(y, t, seq_mask, residue_index))
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    sol = diffeqsolve(term, solver, t0=t_0, t1=t_1, y0=x_0, dt0=dt0,
                        stepsize_controller=stepsize_controller, max_steps=65536)
    
    return sol.ys[-1]

pjit_solve_ode = jax.pmap(jax.jit(solve_ode), axis_name='i', in_axes=(None, None, None, 0, 0, 0))

# Latent Interpolation
if __name__ == "__main__":
    import argparse  # ensure argparse is imported if not already
    parser = argparse.ArgumentParser(description="Script to generate meta stable structure")
    parser.add_argument("--pkl1", type=str,
                        help="Full path for the first pkl file")
    parser.add_argument("--pkl2", type=str,
                        help="Full path for the second pkl file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Directory to save the results")
    # Add new arguments to control interpolation
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for generation")
    parser.add_argument("--interp-method", type=str, default="linear", 
                        choices=["linear", "spherical"],
                        help="Interpolation method: linear or spherical")
    parser.add_argument("--interp-distribution", type=str, default="uniform",
                        choices=["uniform", "gaussian", "logspace"],
                        help="Distribution of interpolation points")
    parser.add_argument("--num-points", type=int, default=None,
                        help="Number of interpolation points (defaults to batch size)")
    args = parser.parse_args()
    
    # Update configuration
    config = ModelConfig(batch_size=args.batch_size, nres=NRES)
    config.set_embedding_dim(protoken_emb.shape[-1], aatype_emb.shape[-1])
    
    # Reset constants
    NRES = config.nres
    DIM_EMB = config.dim_emb
    NDEVICES = config.ndevices
    BATCH_SIZE = config.batch_size
    NSAMPLE_PER_DEVICE = config.nsample_per_device

    pkl_list = [args.pkl1] * (BATCH_SIZE // 2) + [args.pkl2] * (BATCH_SIZE // 2)
    data_dicts = []
    for protoken_file in pkl_list:
        with open(protoken_file, 'rb') as f:
            data = pkl.load(f)
        seq_len = data['seq_len']
        embedding = np.concatenate(
            [protoken_emb[data['protokens'].astype(np.int32)],
             aatype_emb[data['aatype'].astype(np.int32)]], axis=-1)
        embedding = np.pad(embedding, ((0, NRES - seq_len), (0, 0)))
        data_dicts.append({
            'embedding': embedding,
            'seq_mask': np.pad(data['seq_mask'], (0, NRES - seq_len)).astype(np.bool_),
            'residue_index': np.pad(data['residue_index'], (0, NRES - seq_len)).astype(np.int32),
        })
    data_dict = {k: np.stack([d[k] for d in data_dicts], axis=0) for k in data_dicts[0].keys()}

    # For pmap: reshape inputs
    reshape_func = lambda x: x.reshape(NDEVICES, x.shape[0] // NDEVICES, *x.shape[1:])
    data_dict = jax.tree_util.tree_map(reshape_func, data_dict)

    # Forward PF-ODE: data -> latent
    x0 = data_dict['embedding']
    xT = pjit_solve_ode(0, scheduler.num_timesteps, 1.0, x0, data_dict['seq_mask'], data_dict['residue_index'])
    xT_np = np.array(xT)

    # Use the encapsulated interpolation function
    xT = xT.reshape(BATCH_SIZE, NRES, DIM_EMB)
    xT_A, xT_B = xT[0], xT[-1]  # Endpoints
    
    # Use the new interpolation function
    xT_interpolation = interpolate_latent(
        xT_A, xT_B, config, 
        method=args.interp_method,
        distribution=args.interp_distribution,
        num_points=args.num_points
    )

    # Backward PF-ODE: latent -> data
    x0_interpolation = pjit_solve_ode(scheduler.num_timesteps, 0, -1.0, 
                                  xT_interpolation, data_dict['seq_mask'], data_dict['residue_index'])

    ret = {'embedding': x0_interpolation, 'seq_mask': data_dict['seq_mask'], 'residue_index': data_dict['residue_index']}
    ret.update(pjit_index_from_embedding(ret['embedding']))

    ret = jax.tree_util.tree_map(lambda x: np.array(x.reshape(BATCH_SIZE, *x.shape[2:])).tolist(), ret)
    
    os.makedirs(args.output, exist_ok=True)
    result_pkl = os.path.join(args.output, "result.pkl")
    result_flatten_pkl = os.path.join(args.output, "result_flatten.pkl")
    
    with open(result_pkl, 'wb') as f:
        pkl.dump(ret, f)
        
    ret_ = flatten_list_of_dicts([ret])
    with open(result_flatten_pkl, 'wb') as f:
        pkl.dump(ret_, f)

