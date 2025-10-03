#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp

class PolicyNetwork(nn.Module):
    """A network that learns to generate optimal latent space paths."""
    n_points: int  # Number of points in the output path
    n_res: int
    latent_dim: int
    
    def setup(self):
        # A simple MLP to predict the non-linear deviation from a linear path.
        # The network outputs a single vector representing the entire deviation field.
        self.mlp = nn.Sequential([
            nn.Dense(features=1024),
            nn.relu,
            nn.Dense(features=2048),
            nn.relu,
            nn.Dense(features=self.n_points * self.n_res * self.latent_dim)
        ])

    def __call__(self, xT_start, xT_end, key, std_dev=0.1):
        """
        Generates a stochastic path and calculates the log probability of that path.

        Args:
            xT_start: The start latent vector. Shape (n_res, latent_dim)
            xT_end: The end latent vector. Shape (n_res, latent_dim)
            key: JAX random key.
            std_dev: Standard deviation for the Gaussian noise for exploration.

        Returns:
            A tuple of (path, log_prob).
        """
        # 1. Create a baseline linear interpolation path
        lambdas = jnp.linspace(0, 1, self.n_points)[:, None, None]
        linear_path = (1 - lambdas) * xT_start[None, ...] + lambdas * xT_end[None, ...]

        # 2. Use the network to predict a deviation from the linear path
        # Flatten start and end points to create a single input vector
        net_input = jnp.concatenate([xT_start.ravel(), xT_end.ravel()])
        deviation_mean_flat = self.mlp(net_input)
        deviation_mean = deviation_mean_flat.reshape(self.n_points, self.n_res, self.latent_dim)

        # 3. Create a stochastic action by adding noise
        noise = jax.random.normal(key, shape=deviation_mean.shape) * std_dev
        deviation_sample = deviation_mean + noise

        # 4. The final path is the linear base + learned deviation
        # We detach the linear path from the gradient calculation for the deviation
        # as the deviation is what we are learning.
        path = linear_path + deviation_sample

        # 5. Calculate the log probability of the sampled deviation (the action)
        # This is the log-likelihood of a multivariate Gaussian.
        log_prob = -0.5 * jnp.sum((deviation_sample - deviation_mean)**2 / (std_dev**2))
        log_prob -= 0.5 * jnp.log(2 * jnp.pi) * deviation_sample.size
        log_prob -= jnp.log(std_dev) * deviation_sample.size

        return path, log_prob