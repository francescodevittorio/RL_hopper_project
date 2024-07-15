import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb

"""
This script evaluates a trained PPO model on a specified environment.
To test different training-test configurations (source->source, source->target, target->target), 
appropriate changes need to be made to the 'env' and 'model.load()' paths.
"""

def evaluate_trained_model():
    # Initialize Weights & Biases run for logging
    wandb.init(project='baselines', name='source -> source', monitor_gym=True)
    
    # Create the environment and monitor it
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)

    # Load the trained PPO model
    model = PPO.load("./logs/best_model")

    # Evaluate the model on the environment
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    
    # Log the evaluation results to W&B
    wandb.log({
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })

    # Print the mean and standard deviation of the rewards
    print(f"Mean reward: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    # Log in to W&B and start the evaluation process
    wandb.login()
    evaluate_trained_model()
