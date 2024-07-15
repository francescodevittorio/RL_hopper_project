import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import wandb
import torch.nn as nn

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        done = self.locals.get('dones', None)
        if done is not None and done[0]:
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
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)
        self.episode_lengths.append(1)  # Count each step
        return True

def train_best_model():
    wandb.init(project='best_model_training', name='best_model_2', monitor_gym=True)
    
    best_hyperparams = {
    'batch_size': 256,  
    'clip_range': 0.2392019894317692,
    'ent_coef': 0.02724003914136842,
    'gae_lambda': 0.95,  
    'gamma': 0.9917724230758292,
    'learning_rate': 0.00028209230914387576,
    'max_grad_norm': 0.6369742313859069,
    'n_epochs': 10,
    'n_steps': 2048,  
    'policy_kwargs': dict(
        log_std_init=-2,  
        ortho_init=False, 
        activation_fn=nn.LeakyReLU,  
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    ),
    'vf_coef': 0.745648997079238
}

    
    env = gym.make('CustomHopper-target-v0')
    env = Monitor(env)
    eval_env = gym.make('CustomHopper-target-v0')
    eval_env = Monitor(eval_env)

    model = PPO('MlpPolicy', env, **best_hyperparams, verbose=1)
    
    reward_callback = RewardLoggingCallback()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_target2/',
                                 log_path='./logs_target2/', eval_freq=5000, n_eval_episodes=50,
                                 deterministic=True, render=False)
    
    model.learn(total_timesteps=3000000, callback=[reward_callback, eval_callback])  

    model.save("bestPPO_model_target_last_2")
    wandb.save("bestPPO_model_target_last_2.zip")

if __name__ == '__main__':
    wandb.login()
    train_best_model()
