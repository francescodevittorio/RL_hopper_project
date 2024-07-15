import gym
from env.custom_hopper_UDR import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import wandb
import torch.nn as nn

"""
This script trains a PPO model using Uniform Domain Randomization.
The linked environment is 'custom_hopper_UDR.py'.
We have implemented different levels of randomization: 10%, 20%, 30%, 40%, and 50%.
To switch between these implementations, appropriate modifications need to be made in the environment creation and in some lines of this script.
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
        self.episode_lengths.append(1)  # Count each step
        return True

# Function to train the UDR PPO model
def train_best_model():
    wandb.init(project='UDR_train', name='UDR_50%', monitor_gym=True)
    
    # Define the best hyperparameters found from previous experiments
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
    
    # Create and monitor the source environment
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)
    # Create and monitor the target environment for evaluation
    eval_env = gym.make('CustomHopper-target-v0')
    eval_env = Monitor(eval_env)

    # Initialize the PPO model with the best hyperparameters
    model = PPO('MlpPolicy', env, **best_hyperparams, verbose=1)
    
    reward_callback = RewardLoggingCallback()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_UDR_50%/',
                                 log_path='./logs_UDR_50%/', eval_freq=5000, n_eval_episodes=50,
                                 deterministic=True, render=False)
    
    # Train the model
    model.learn(total_timesteps=3000000, callback=[reward_callback, eval_callback])  

    # Save the trained model
    model.save("UDR_50%")
    wandb.save("UDR_50%")

if __name__ == '__main__':
    wandb.login()
    train_best_model()

