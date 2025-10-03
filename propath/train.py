#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import our custom modules
from propath.rl_path_finder import ProTokenWrapper, PathEvaluator, ProteinPathEnv
from propath.agent import Agent

# JAX/Flax imports for training
import optax

def train(args):
    """Main training loop."""

    # 1. Initialize Environment and Agent
    print("Initializing environment, wrapper, and agent...")
    wrapper = ProTokenWrapper(
        ckpt_path=args.ckpt,
        encoder_config_path=args.encoder_config,
        decoder_config_path=args.decoder_config,
        vq_config_path=args.vq_config,
        padding_len=args.padding_len
    )
    evaluator = PathEvaluator()
    env = ProteinPathEnv(wrapper, evaluator, args.start_pdb, args.end_pdb, max_steps=args.max_steps)

    # Determine state and action dimensions from the environment embeddings
    state_dim = env.start_emb.shape[-1] * 2 # Concatenation of current and target embeddings
    action_dim = env.start_emb.shape[-1]
    
    agent = Agent(state_dim=state_dim, action_dim=action_dim, learning_rate=args.learning_rate)

    # 2. Define the training step function (using JAX)
    @jax.jit
    def train_step(train_state, trajectory):
        """Performs a single update step of the agent's policy network."""
        
        def calculate_loss(params):
            # Unpack trajectory
            states, actions, rewards = trajectory
            
            # Calculate discounted returns (Gamma)
            discounts = jnp.power(args.gamma, jnp.arange(len(rewards)))
            returns = jnp.cumsum(rewards[::-1] * discounts[::-1])[::-1] / discounts
            
            # Get the actions predicted by the policy for the states in the trajectory
            predicted_actions = train_state.apply_fn({'params': params}, states)
            
            # REINFORCE loss: -log_prob(action) * return
            # For a deterministic policy with continuous actions, a simple MSE loss
            # against the action taken, weighted by the return, can be a starting point.
            # This is a simplification; more advanced algorithms would use log probabilities of a distribution.
            loss = jnp.mean(((predicted_actions - actions)**2) * returns[:, None])
            return loss

        grad_fn = jax.value_and_grad(calculate_loss)
        loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    # 3. Main Training Loop
    print("Starting training...")
    episode_rewards = []

    for episode in tqdm(range(args.num_episodes)):
        # Collect a trajectory
        states, actions, rewards = [], [], []
        state = env.reset()
        
        for step in range(env.max_steps):
            # Prepare state for the agent (concatenate current and target embeddings)
            agent_state = jnp.concatenate([state[0], state[1]], axis=-1)
            
            # Select action
            action = agent.select_action(agent.train_state.params, agent_state)
            # In a real scenario, add noise for exploration
            # action += jax.random.normal(agent.key, shape=action.shape) * args.exploration_noise
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            states.append(agent_state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # Prepare trajectory for training
        trajectory = (jnp.array(states), jnp.array(actions), jnp.array(rewards))
        
        # Update the agent
        agent.train_state, loss = train_step(agent.train_state, trajectory)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if episode % args.log_interval == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Loss: {loss:.4f}")

    print("Training finished.")
    # Here you would save the trained agent parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for RL-based protein path finding.')
    
    # Structure inputs
    parser.add_argument('--start_pdb', type=str, required=True, help='Path to the starting PDB file.')
    parser.add_argument('--end_pdb', type=str, required=True, help='Path to the ending PDB file.')

    # Model inputs
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pkl file).')
    parser.add_argument('--encoder_config', type=str, required=True, help='Path to the encoder config yaml.')
    parser.add_argument('--decoder_config', type=str, required=True, help='Path to the decoder config yaml.')
    parser.add_argument('--vq_config', type=str, required=True, help='Path to the VQ config yaml.')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the agent.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards.')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for printing training logs.')
    parser.add_argument('--padding_len', type=int, default=768, help="Padding length for the model.")

    args = parser.parse_args()
    
    train(args)
