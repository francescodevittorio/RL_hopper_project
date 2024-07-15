import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb

"""
This script evaluates a PPO model trained with Uniform Domain Randomization.
The linked environment is 'custom_hopper_UDR.py'.
We have implemented different levels of randomization: 10%, 20%, 30%, 40%, and 50%.
To switch between these implementations, appropriate modifications need to be made in the environment creation and in some lines of this script.
"""

def evaluate_trained_model():
    # Initialize a W&B run for logging purposes
    wandb.init(project='UDR_test_source', name='UDR_30%_source', monitor_gym=True)
    
    # Create and monitor the source environment
    env = gym.make('CustomHopper-source-v0')
    env = Monitor(env)

    # Load the trained PPO model from the specified path
    model = PPO.load("./logs_UDR_30%/best_model")

    # Evaluate the model on the environment for a specified number of episodes
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    
    # Log the evaluation results to W&B
    wandb.log({
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })

    # Print the evaluation results
    print(f"Mean reward: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    # Log in to W&B and start the evaluation process
    wandb.login()
    evaluate_trained_model()

