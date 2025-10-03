#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Sequence

class PolicyNetwork(nn.Module):
    """
    A simple MLP policy network.
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: A JAX array representing the concatenated current and target embeddings.
        
        Returns:
            An action vector of shape (action_dim,).
        """
        x = state
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.relu(x)
        
        # Output layer for the action
        action = nn.Dense(features=self.action_dim)(x)
        return action

class Agent:
    """
    The RL Agent that contains the policy network and handles updates.
    """
    def __init__(self, state_dim: int, action_dim: int, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy = PolicyNetwork(action_dim=action_dim)
        
        # Initialize the training state
        self.key = jax.random.PRNGKey(0)
        dummy_state = jnp.zeros((1, self.state_dim))
        params = self.policy.init(self.key, dummy_state)['params']
        
        optimizer = optax.adam(learning_rate)
        
        self.train_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=params,
            tx=optimizer
        )

    @jax.jit
    def select_action(self, params, state):
        """
        Selects an action based on the current policy and state.
        For now, this is deterministic. We can add noise for exploration later.
        """
        # Add batch dimension if missing
        if state.ndim == 1:
            state = state[jnp.newaxis, :]
            
        action = self.train_state.apply_fn({'params': params}, state)
        
        # Remove batch dimension for single action
        return action.squeeze(0)

    @jax.jit
    def update(self, state, trajectory):
        """
        Updates the policy network using a simple policy gradient method (REINFORCE).
        This is a placeholder for the actual training logic which will be in train.py
        """
        # This function will be more complex in the actual training script.
        # For now, it's a placeholder to show the structure.
        # The actual implementation will compute gradients and apply them.
        print("Agent.update() called. Actual update logic will be in train.py")
        return self.train_state

