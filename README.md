# RL_project_hopper
This project implements various reinforcement learning algorithms for training a CustomHopper environment. The project includes training, hyperparameter optimization, and evaluation scripts.

## Scripts Description

### `agent_Actor_Critic.py`

This script defines the policy, critic, and agent classes for the Actor-Critic algorithm. 
The linked train script is: `train_Actor_Critic.py`.

### `agent_Reinforce_Baseline.py`

This script defines the policy and agent classes for the REINFORCE with baseline algorithm.
The linked train script is: `train_Reinforce_Baseline.py`.

### `agent_Reinforce.py`

This script defines the policy and agent classes for the REINFORCE algorithm.
The linked train script is: `train_Reinforce.py`.

### `train_Actor_Critic.py`

This script trains an agent using the Actor-Critic algorithm on the CustomHopper environment. 
The linked script of the agent is: `agent_Actor_Critic.py`.

### `train_Reinforce_Baseline.py`

This script trains an agent using the REINFORCE with baseline algorithm on the CustomHopper environment. 
The linked script of the agent is: `agent_Reinforce_Baseline.py`.

### `train_Reinforce.py`

This script trains an agent using the REINFORCE algorithm on the CustomHopper environment. 
The linked script of the agent is: `agent_Reinforce.py`.

### `optimize_PPO.py`

This script performs hyperparameter optimization for the PPO algorithm.
It uses Weights & Biases (wandb) for tracking experiments and performing Bayesian optimization of hyperparameters. 
The script defines a custom callback for logging rewards and integrates with stable-baselines3 for training the PPO model.

### `PPO_best_source_train.py`

This script trains a PPO model, with the best hyperparameter combination, on the source environment.
It uses Stable-Baselines3 for the PPO implementation and integrates Weights & Biases for logging.
The training is performed on the 'CustomHopper-source-v0' environment, and the evaluation is done on the 'CustomHopper-target-v0' environment.

### `PPO_best_target_train.py`

This script trains a PPO model with the best hyperparameter combination on the target environment.
It uses Stable-Baselines3 for the PPO implementation and integrates Weights & Biases for logging.
The training and evaluation are both performed on the 'CustomHopper-target-v0' environment.

### `PPO_test.py`

This script evaluates a trained PPO model on a specified environment.
To test different training-test configurations (source->source, source->target, target->target), 
appropriate changes need to be made to the 'env' and 'model.load()' paths.

### `UDR_train.py`

This script trains a PPO model using Uniform Domain Randomization.
The linked environment is 'custom_hopper_UDR.py'.
We have implemented different levels of randomization: 10%, 20%, 30%, 40%, and 50%.
To switch between these implementations, appropriate modifications need to be made in the environment creation and in some lines of this script.

### `UDR_test.py`

This script evaluates a PPO model trained with Uniform Domain Randomization.
The linked environment is 'custom_hopper_UDR.py'.
We have implemented different levels of randomization: 10%, 20%, 30%, 40%, and 50%.
To switch between these implementations, appropriate modifications need to be made in the environment creation and in some lines of this script.

### `safe_PPO.py`

This script trains a safe PPO policy in the to ensure safety during initial real-world trajectory collection. 
The policy is trained with adjusted reward weights to prioritize the robot's safety, primarily focusing on maintaining the robot's balance 
and stability over achieving high speeds. The linked environment is 'custom_hopper_safe.py'.
This safe policy will later be used as the initial policy in the SimOpt with GANs algorithm.

### `gridsearch_project_extension.py`

This script performs a grid search to optimize the learing rates of discriminator and generator and the simulated batch size of the simulated trajectories for the SimOpt algorithm with GANs.
The results of the grid search are logged to CSV files and visualized using Weights and Biases.

### `project_extension.py`

This script implements our project extension. It involves training a generator and discriminator 
network using the SimOpt algorithm to optimize the dynamics parameters of the CustomHopper environment.
This script requires modifications for different runs, such as changing hyperparameters, seed or iteration counts.

## Requirements

Install the required packages using `requirements.txt`.
