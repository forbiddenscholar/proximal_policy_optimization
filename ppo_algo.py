from torch.distributions import Categorical
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as categorical

class PPO_memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.values = []
        self.dones = []
        self.log_probs = []
        self.rewards = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, value, reward, log_prob, done):
        self.states.append(state)
        self.values.append(value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                    fc1_dims=256, fc2_dims=256, chkpt_dir='./chkpts/actor'):
        super(ActorNetwork, self).__init__()

        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'ppo_actor')

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))