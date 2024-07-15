import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import wandb
import torch.nn as nn

"""
This script trains a PPO model, with the best hyperparameter combination, on the source environment.
It uses Stable-Baselines3 for the PPO implementation and integrates Weights & Biases for logging.
The training is performed on the 'CustomHopper-source-v0' environment, and the evaluation is done on the 'CustomHopper-target-v0' environment.
"""

# Callback class for logging rewards to Weights & Biases 
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    # This function is called at each step
    def _on_step(self) -> bool:
        done = self.locals.get('dones', None)
        if done is not None and done[0]:
            # Log episode reward and length when an episode is done
            episode_reward = sum(self.episode_rewards)
            episode_length = len(self.episode_lengths)
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_count += 1
            wandb.log({
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "episode_count": self.episode_count
            })
        # Append reward for the current step
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)
        self.episode_lengths.append(1)  
        return True

# Function to train the PPO model with the best hyperparameters
def train_best_model():
    # Initialize W&B run
    wandb.init(project='best_model_training', name='source_train', monitor_gym=True)
    
    # Best hyperparameters obtained from optimization
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
    
    # Create the training environment and monitor it
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)
    # Create the evaluation environment and monitor it
    eval_env = gym.make('CustomHopper-target-v0')
    eval_env = Monitor(eval_env)

    # Create the PPO model with the specified hyperparameters
    model = PPO('MlpPolicy', env, **best_hyperparams, verbose=1)
    
    # Set up reward logging and evaluation callbacks
    reward_callback = RewardLoggingCallback()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000, n_eval_episodes=50,
                                 deterministic=True, render=False)
    
    # Train the model for a total of 3,000,000 timesteps
    model.learn(total_timesteps=3000000, callback=[reward_callback, eval_callback])  

    # Save the trained model locally and upload to W&B
    model.save("bestPPO_model_source")
    wandb.save("bestPPO_model_source.zip")

if __name__ == '__main__':
    # Log in to W&B and start the training process
    wandb.login()
    train_best_model()
