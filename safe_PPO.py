import gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from env.custom_hopper_safe import *
import wandb

"""
This script trains a safe PPO policy in the to ensure safety during initial real-world trajectory collection. 
The policy is trained with adjusted reward weights to prioritize the robot's safety, primarily focusing on maintaining the robot's balance 
and stability over achieving high speeds. The linked environment is 'custom_hopper_safe.py'.
This safe policy will later be used as the initial policy in the SimOpt with GANs algorithm.
"""

# Definition of the best hyperparameters
best_hyperparams = {
    'batch_size': 256,
    'clip_range': 0.273070939364655,
    'ent_coef': 0.021034494707570742,
    'gae_lambda': 0.95,
    'gamma': 0.994413048323934,
    'learning_rate': 0.0009816214750301055,
    'max_grad_norm': 0.4268132364413415,
    'n_epochs': 20,
    'n_steps': 2048,
    'vf_coef': 0.7511263756793052
}

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self):
        if self.locals['dones'][0]:
            episode_rewards = self.locals['infos'][0]['episode']['r']
            episode_length = self.locals['infos'][0]['episode']['l']
            wandb.log({
                "episode_reward": episode_rewards,
                "episode_length": episode_length
            })
        return True

# Initialize the environment
env = gym.make('CustomHopper-source-safe-v0')
eval_env = gym.make('CustomHopper-source-safe-v0')

# Initialize wandb
wandb.init(project="safe-hopper", config=best_hyperparams)

# Create the PPO model with the best hyperparameters
model = PPO(
    'MlpPolicy',
    eval_env,
    batch_size=best_hyperparams['batch_size'],
    clip_range=best_hyperparams['clip_range'],
    ent_coef=best_hyperparams['ent_coef'],
    gae_lambda=best_hyperparams['gae_lambda'],
    gamma=best_hyperparams['gamma'],
    learning_rate=best_hyperparams['learning_rate'],
    max_grad_norm=best_hyperparams['max_grad_norm'],
    n_epochs=best_hyperparams['n_epochs'],
    n_steps=best_hyperparams['n_steps'],
    vf_coef=best_hyperparams['vf_coef'],
    verbose=1
)

# Define the wandb callback
wandb_callback = WandbCallback()

# Define the evaluation callback for model evaluation and saving
eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs_safe/',
    log_path='./logs_safe/',
    eval_freq=5000,  # Evaluation frequency in timesteps
    deterministic=True,
    render=False
)

# Train the model
model.learn(total_timesteps=1000000, callback=[wandb_callback, eval_callback])

# Save the final model
model.save("safe_model")

# End the wandb run
wandb.finish()

