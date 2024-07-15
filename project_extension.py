import torch
from torch import nn, optim
import numpy as np
import csv
import wandb
import gym
import os
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

"""
This script implements our project extension. It involves training a generator and discriminator 
network using the SimOpt algorithm to optimize the dynamics parameters of the CustomHopper environment.
This script requires modifications for different runs, such as changing hyperparameters, seed or iteration counts. 
"""

# Set the seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim, param_dim, output_range):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, param_dim)
        )
        self.output_range = output_range
        self._initialize_weights()  # Initialize weights once here

    def forward(self, x):
        x = self.network(x)
        x = torch.tanh(x)
        min_val, max_val = self.output_range
        x = min_val + (max_val - min_val) * (x + 1) / 2  # Rescale from (-1, 1) to (min_val, max_val)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, observation_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(observation_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# Callback for wandb logging
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

# Function to apply parameters to the environment
# This function modifies the environment's parameters with the generated values.
def apply_parameters_to_env(env, params):
    env.sim.model.body_mass[2:] = params

# Function to simulate the environment
# This function runs the environment with a given policy model and returns the observed states and total reward.
def simulate(env, model):
    observations = []
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        total_reward += reward
    
    observations = np.array(observations)
    return torch.tensor(observations, dtype=torch.float32), total_reward

# Function to pad trajectories
# This function pads the trajectories to ensure they all have the same length for processing in batches.
def pad_trajectories(trajectories, observation_dim):
    max_length = max(len(traj) for traj in trajectories)
    padded_trajectories = torch.zeros((len(trajectories), max_length, observation_dim))
    
    for i, traj in enumerate(trajectories):
        length = len(traj)
        padded_trajectories[i, :length] = torch.tensor(traj, dtype=torch.float32)
    
    return padded_trajectories

# Fixed parameters
latent_dim = 10  # Dimensionality of the latent space for the generator
param_dim = 3  # Number of parameters to be generated
observation_dim = 11  # Dimensionality of the observations
output_range = (0.5, 6)  # Range of the output parameters
num_epochs = 12  # Number of epochs for training in each simopt iteration
num_simopt_iterations = 5  # Number of simopt iterations
real_batch_size = 16  # Batch size for real trajectories
simulated_batch_size = 64  # Batch size for simulated trajectories
num_env = 4  # Number of sets of parameters per each epoch

# Initialize environments
source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')

# Initialize wandb for logging
wandb.init(project="extension")

# Load the safe model
model = PPO.load("./logs_safe/safe_model.zip")

# Initialize generator and discriminator
generator = Generator(latent_dim, param_dim, output_range)
discriminator = Discriminator(observation_dim)
loss_function = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=1e-3)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-2)

# Open CSV files to write results
with open('generated_params_finals.csv', mode='w', newline='') as params_file, \
     open('simulated_results_finals.csv', mode='w', newline='') as simulated_file, \
     open('real_results_finals.csv', mode='w', newline='') as real_file:

    params_writer = csv.writer(params_file)
    simulated_writer = csv.writer(simulated_file)
    real_writer = csv.writer(real_file)

    # Write headers to CSV files
    params_writer.writerow(["simopt_iteration", "epoch", "params"])
    simulated_writer.writerow(["simopt_iteration", "epoch", "mean_simulated_reward", "std_simulated_reward", "d_loss_real", "d_loss_fake", "d_loss", "g_loss"])
    real_writer.writerow(["simopt_iteration", "mean_real_reward", "std_real_reward"])

    for simopt_iteration in range(num_simopt_iterations):
        real_trajectories = []
        real_rewards = []
        for _ in range(real_batch_size):
            traj, reward = simulate(target_env, model)
            real_trajectories.append(traj)
            real_rewards.append(reward)

        for epoch in range(num_epochs):
            # Define the number of timesteps based on the epoch
            if epoch < 3:
                timesteps = 10000
            elif epoch < 6:
                timesteps = 30000
            elif epoch < 9:
                timesteps = 60000
            else:
                timesteps = 100000

            z = torch.randn(num_env, latent_dim)
            generated_params = generator(z)

            for params in generated_params:
                apply_parameters_to_env(source_env, params.detach().numpy())
                model.set_env(source_env)
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=WandbCallback())

            # Collect simulated trajectories from the source environment
            simulated_trajectories = []
            simulated_rewards = []
            for _ in range(simulated_batch_size):
                traj, reward = simulate(source_env, model)
                simulated_trajectories.append(traj)
                simulated_rewards.append(reward)

            # Prepare real trajectories for discriminator training
            real_trajectories_tensor = pad_trajectories(real_trajectories, observation_dim)
            real_labels = torch.ones(real_batch_size, 1)  # Labels for real trajectories are set to 1
            real_outputs = discriminator(real_trajectories_tensor)
            d_loss_real = loss_function(real_outputs, real_labels)  # Compute the loss for real trajectories

            # Prepare simulated trajectories for discriminator training
            simulated_trajectories_tensor = pad_trajectories(simulated_trajectories, observation_dim)
            fake_labels = torch.zeros(simulated_batch_size, 1)  # Labels for fake trajectories are set to 0
            fake_outputs = discriminator(simulated_trajectories_tensor)
            d_loss_fake = loss_function(fake_outputs, fake_labels)  # Compute the loss for simulated trajectories

            # Update the discriminator
            optimizer_D.zero_grad()  # Zero the gradients for the discriminator optimizer
            # Compute the weighted average of the real and fake losses
            real_weight = simulated_batch_size / (real_batch_size + simulated_batch_size)
            simulated_weight = real_batch_size / (real_batch_size + simulated_batch_size)
            d_loss = d_loss_real * real_weight + d_loss_fake * simulated_weight
            d_loss.backward()  # Backpropagate the loss
            optimizer_D.step()  # Update the discriminator weights

            # Update the generator
            optimizer_G.zero_grad()  # Zero the gradients for the generator optimizer
            # Compute the loss for the generator
            fake_outputs_for_generator = discriminator(simulated_trajectories_tensor)
            g_loss = loss_function(fake_outputs_for_generator, torch.ones(simulated_batch_size, 1))  # Labels are set to 1 for generator training
            g_loss.backward()  # Backpropagate the loss
            optimizer_G.step()  # Update the generator weights

            params_log = [param.detach().cpu().numpy().tolist() for param in generated_params]

            # Write generated parameters to the CSV file
            for param_set in params_log:
                params_writer.writerow([simopt_iteration + 1, epoch + 1, param_set])
                params_file.flush()  # Ensure data is written immediately

            mean_simulated_reward = np.mean(simulated_rewards)
            std_simulated_reward = np.std(simulated_rewards)

            simulated_writer.writerow([simopt_iteration + 1, epoch + 1, mean_simulated_reward, std_simulated_reward, d_loss_real.item(), d_loss_fake.item(), d_loss.item(), g_loss.item()])
            simulated_file.flush()  # Ensure data is written immediately

            # Log losses and simulated rewards with epoch progression
            wandb.log({
                "epoch": epoch + 1,
                "d_loss_real": d_loss_real.item(),
                "d_loss_fake": d_loss_fake.item(),
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "simulated_rewards_mean": mean_simulated_reward,
                "simulated_rewards_std": std_simulated_reward
            })

            # Save the PPO model after each epoch with the epoch number in the file name
            checkpoint_path = os.path.join("extension", f"sim_opt_model_simopt_{simopt_iteration}_epoch_{epoch}.zip")
            model.save(checkpoint_path)

        mean_real_reward = np.mean(real_rewards)
        std_real_reward = np.std(real_rewards)
        real_writer.writerow([simopt_iteration + 1, mean_real_reward, std_real_reward])
        real_file.flush()  # Ensure data is written immediately

        # Log real rewards with simopt iterations progression
        wandb.log({
            "simopt_iteration": simopt_iteration + 1,
            "real_rewards_mean": mean_real_reward,
            "real_rewards_std": std_real_reward
        })

wandb.finish()
