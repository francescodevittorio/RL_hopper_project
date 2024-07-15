import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

"""
This script defines the policy and agent classes for the REINFORCE algorithm. 
The linked train script is: 'train_Reinforce.py'.
"""

# calculate discounted rewards
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

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
        self.sigma_activation = F.softplus # ensures that the variance is always positive
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    # initialize the weights of the network with a normal distribution, while initializing the biases to zero
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    # returns a multivariate normal distribution over the action space
    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99 # discount factor
        self.action_log_probs = [] # log probabilities of the actions of the episode
        self.rewards = [] # store of the episode rewards

    def update_policy(self):
        # Convert lists of log probabilities and rewards to tensors and move them to the specified device
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        # compute discounted returns
        discounted_returns = discount_rewards(rewards, self.gamma)
        # compute policy gradient loss function given log probability and discounted returns
        policy_loss = -torch.mean(action_log_probs * discounted_returns)
        
        # compute gradients
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # step the optimizer
        self.optimizer.step()

        return        

    # function that returns the next action given the current state, with an option to specify if we are evaluating or training the policy
    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, action_log_prob, reward):
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))

