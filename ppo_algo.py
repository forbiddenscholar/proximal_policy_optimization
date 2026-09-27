import torch
import numpy as np
import os
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