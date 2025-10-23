#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Sequence

class PolicyNetwork(nn.Module):
    """
    A stochastic policy network that outputs parameters for a Gaussian distribution.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: A JAX array representing the concatenated current and target embeddings.
        
        Returns:
            A tuple containing the mean and log_std of the action distribution.
        """
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.relu(x)
        
        # Output layer for the mean of the action
        mean = nn.Dense(features=self.action_dim)(x)
        
        # Output layer for the log standard deviation of the action
        log_std = nn.Dense(features=self.action_dim)(x)
        
        return mean, log_std

import jax.random as random
from jax.scipy.stats import norm

class Agent:
    """
    The RL Agent that contains the policy network and handles updates.
    """
    def __init__(self, state_dim: int, action_dim: int, learning_rate=1e-4, seed=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy = PolicyNetwork(action_dim=action_dim)
        
        # Initialize the training state
        self.key = random.PRNGKey(seed)
        dummy_state = jnp.zeros((1, self.state_dim))
        params = self.policy.init(self.key, dummy_state)['params']
        
        optimizer = optax.adam(learning_rate)
        
        self.train_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=params,
            tx=optimizer
        )

    @jax.jit
    def select_action(self, params, state, key):
        """
        Selects an action by sampling from the policy's output distribution.
        Returns the action and its log probability.
        """
        # Add batch dimension if missing
        if state.ndim == 1:
            state = state[jnp.newaxis, :]
            
        mean, log_std = self.train_state.apply_fn({'params': params}, state)
        std = jnp.exp(log_std)
        
        # Sample action from the Gaussian distribution
        action = mean + std * random.normal(key, shape=mean.shape)
        
        # Calculate the log probability of the action
        log_prob = jnp.sum(norm.logpdf(action, loc=mean, scale=std), axis=-1)
        
        # Remove batch dimension for single action
        return action.squeeze(0), log_prob.squeeze(0)


