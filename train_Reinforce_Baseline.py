import argparse
import time
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from env.custom_hopper import *
from agent_Reinforce_Baseline import Agent, Policy

"""
This script trains an agent using the REINFORCE with baseline algorithm on the CustomHopper environment. 
The linked script of the agent is: `agent_Reinforce_Baseline.py`.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=500000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    return parser.parse_args()

args = parse_args()

# Function to set the random seed for reproducibility
def set_seed(seed, env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def main():
    start_time = time.time()

    # Create the environment
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    # Set seed for reproducibility
    set_seed(args.seed, env)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    episode_rewards = []
    episode_lengths = []

    # Main training loop
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        episode_length = 0
        state = env.reset()  # Reset the environment and observe the initial state

        # Reset agent's log probabilities and rewards for the new episode
        agent.action_log_probs = []
        agent.rewards = []

        # Loop until the episode is done
        while not done:
            # Get action and log probabilities from the policy
            action, log_action_probabilities = agent.get_action(state)

            # Perform the action in the environment
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            # Store the outcome of the action
            agent.store_outcome(log_action_probabilities, reward)
            
            # Increment the episode length and accumulate the reward
            episode_length += 1
            train_reward += reward

        # Update the policy after the episode ends
        agent.update_policy()

        # Store the total reward and length of the episode
        episode_rewards.append(train_reward)
        episode_lengths.append(episode_length)

        # Print progress every 'print_every' episodes
        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode + 1)
            print('Episode return:', train_reward)
    
    # Save the model parameters
    torch.save(agent.policy.state_dict(), "model.mdlTask2Baseline")

    # Create results directory if it doesn't exist
    if not os.path.exists('results_baseline'):
        os.makedirs('results_baseline')

    # Plot and save reward per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.n_episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig('results_baseline/reward_per_episode.png')
    plt.close()

    # Plot and save episode lengths
    plt.figure(figsize=(12, 6))
    plt.plot(range(args.n_episodes), episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.title('Episode length per Episode')
    plt.savefig('results_baseline/episode_length_per_episode.png')
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total execution time: {total_time:.2f} seconds')

if __name__ == '__main__':
    main()
