import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

"""" 
This script defines the policy, critic, and agent classes for the Actor-Critic algorithm. 
The linked train script is: `train_Actor_Critic.py`.
"""

# class that implements the policy
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        
        # policy network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus  # ensures variance is always positive
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    # Initialize weights of the network with a normal distribution, set biases to zero
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    # Returns a multivariate normal distribution over the action space
    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist

class Critic(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # critic network
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    # Initialize weights of the network with a normal distribution, set biases to zero
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value_estimate = self.fc3_critic(x_critic)
        return value_estimate.T

class Agent(object):
    def __init__(self, policy, critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        self.gamma = 0.99  # discount factor
        self.states = []  # store states
        self.next_states = []  # store next states
        self.action_log_probs = []  # log probabilities of the actions of the episode
        self.rewards = []  # store of the episode rewards
        self.done = []  # store done flags

    def update_policy_critic(self):
        # Convert lists to tensors and move them to the appropriate device
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        # Compute value estimates for the next states
        # Detach the tensor from the computation graph to avoid backpropagating through the critic during the next state value computation
        values_next_states = self.critic(next_states).detach()
        
        # Compute bootstrapped estimates of state values using rewards and the value of next states
        # (1 - done) ensures that we do not use bootstrapping if the episode has ended
        bootstrapped_estimates = rewards + self.gamma * values_next_states * (1 - done)
        
        # Compute value estimates for the current states
        values_states = self.critic(states)
        
        # Compute advantages: how much better the taken actions are compared to the expected value
        # Detach values_states to avoid backpropagating through the critic twice
        advantages = bootstrapped_estimates - values_states.detach()
            
        # Compute policy loss
        # Negative sign because we perform gradient ascent (maximize) on log-probabilities weighted by advantages
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Update the policy network
        self.policy_optimizer.zero_grad()  # Zero the gradients
        policy_loss.backward()  # Backpropagate the loss
        self.policy_optimizer.step()  # Update the network weights
        
        # Compute critic loss as the Mean Squared Error between the predicted values and bootstrapped estimates
        critic_loss = F.mse_loss(self.critic(states), bootstrapped_estimates)
        
        # Update the critic network
        self.critic_optimizer.zero_grad()  # Zero the gradients
        critic_loss.backward()  # Backpropagate the loss
        self.critic_optimizer.step()  # Update the network weights


    # Function that returns the next action given the current state, with an option to evaluate or train the policy
    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)  

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    # Store the outcome of an action
    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
