import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn

"""
This script performs hyperparameter optimization for the PPO algorithm. 
It uses Weights & Biases (wandb) for tracking experiments and performing Bayesian optimization of hyperparameters. 
The script defines a custom callback for logging rewards and integrates with stable-baselines3 for training the PPO model.
"""

# Custom callback for logging rewards to wandb
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        done = self.locals.get('dones', None)
        if done is not None and done[0]:
            episode_reward = sum(self.episode_rewards)
            episode_length = len(self.episode_rewards)
            self.episode_rewards = []
            wandb.log({
                "episode_reward": episode_reward,
                "episode_length": episode_length
            })
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)
        return True

# Function to train the PPO model
def train_model():
    # Initialize wandb for logging
    wandb.init(monitor_gym=False)
    
    # Function to retrieve hyperparameters from wandb configuration
    def optimize_ppo():
        return {
            'n_steps': wandb.config.n_steps,
            'gamma': wandb.config.gamma,
            'learning_rate': wandb.config.learning_rate,
            'ent_coef': wandb.config.ent_coef,
            'clip_range': wandb.config.clip_range,
            'n_epochs': wandb.config.n_epochs,
            'max_grad_norm': wandb.config.max_grad_norm,
            'batch_size': wandb.config.batch_size,
            'vf_coef': wandb.config.vf_coef,
        }

    # Retrieve hyperparameters
    params = optimize_ppo()

    # Create and monitor the environment
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)

    # Set up callbacks for logging rewards and integrating with wandb
    reward_callback = RewardLoggingCallback()
    wandb_callback = WandbCallback()
    
    # Initialize the PPO model with the optimized hyperparameters
    model = PPO('MlpPolicy', env, **params, verbose=1)

    # Train the model
    model.learn(total_timesteps=1000000, callback=[reward_callback, wandb_callback])

    # Evaluate the trained model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=30)
    
    # Log the mean reward
    wandb.log({'mean_reward': mean_reward})

    return mean_reward

if __name__ == '__main__':
    wandb.login()

    # Define the hyperparameter sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'n_steps': {'values': [512, 1024, 2048]},
            'gamma': {'min': 0.99, 'max': 0.999},
            'learning_rate': {'min': 1e-5, 'max': 1e-3},
            'ent_coef': {'min': 0.0005, 'max': 0.05},
            'clip_range': {'min': 0.1, 'max': 0.4},
            'n_epochs': {'values': [10, 20]},
            'max_grad_norm': {'min': 0.2, 'max': 0.8},
            'batch_size': {'values': [64, 128, 256]},
            'vf_coef': {'min': 0.7, 'max': 0.9}
        }
    }

    # Initialize the hyperparameter sweep
    sweep_id = wandb.sweep(sweep_config, project='hyperparameter_tuning')

    # Run the sweep agent to perform the hyperparameter optimization
    wandb.agent(sweep_id, function=train_model, count=20)
