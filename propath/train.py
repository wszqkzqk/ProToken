#!/usr/bin/env python3

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
sys.path.append('..')

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

import jax.random as random

def train(args):
    """Main training loop."""

    # 1. Initialize Environment and Agent
    print("Initializing environment, wrapper, and agent...")
    # ... (rest of initialization is the same)
    wrapper = ProTokenWrapper(
        ckpt_path=args.ckpt,
        encoder_config_path=args.encoder_config,
        decoder_config_path=args.decoder_config,
        vq_config_path=args.vq_config,
        padding_len=args.padding_len
    )
    evaluator = PathEvaluator()
    env = ProteinPathEnv(wrapper, evaluator, args.start_pdb, args.end_pdb, max_steps=args.max_steps)

    state_dim = env.start_emb.shape[-1] * 2
    action_dim = env.start_emb.shape[-1]
    
    agent = Agent(state_dim=state_dim, action_dim=action_dim, learning_rate=args.learning_rate)
    main_key = random.PRNGKey(args.seed)

    # 2. Define the training step function
    @jax.jit
    def train_step(train_state, trajectory):
        """Performs a single update step using the REINFORCE algorithm."""
        
        def calculate_loss(params):
            states, log_probs, rewards = trajectory
            
            # Calculate discounted returns
            discounts = jnp.power(args.gamma, jnp.arange(len(rewards)))
            returns = jnp.cumsum(rewards[::-1] * discounts[::-1])[::-1] / discounts
            
            # Normalize returns for stability
            returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
            
            # REINFORCE loss: - (log_prob * discounted_return)
            policy_loss = -jnp.mean(log_probs * returns)
            return policy_loss

        grad_fn = jax.value_and_grad(calculate_loss)
        loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    # 3. Main Training Loop
    print("Starting training...")
    episode_rewards = []

    for episode in tqdm(range(args.num_episodes)):
        # Collect a trajectory
        states, log_probs, rewards = [], [], []
        state = env.reset()
        
        # Split key for the episode
        main_key, episode_key = random.split(main_key)

        for step in range(env.max_steps):
            # Prepare state for the agent
            agent_state = jnp.concatenate([state[0], state[1]], axis=-1)
            
            # Split key for action selection
            episode_key, action_key = random.split(episode_key)
            
            # Select action
            action, log_prob = agent.select_action(agent.train_state.params, agent_state, action_key)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            states.append(agent_state)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        # Prepare trajectory for training
        trajectory = (jnp.array(states), jnp.array(log_probs), jnp.array(rewards))
        
        # Update the agent
        agent.train_state, loss = train_step(agent.train_state, trajectory)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if episode % args.log_interval == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Loss: {loss:.4f}")

    print("Training finished.")


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
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--padding_len', type=int, default=768, help="Padding length for the model.")

    args = parser.parse_args()
    
    train(args)
