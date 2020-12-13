import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sample_ind = np.random.choice(len(self.memory), self.batch_size)
        # get the selected experiences: avoid using mid list indexing
        es, ea, er, en, ed = [], [], [], [], []
        i = 0
        while i < len(sample_ind):
            self.memory.rotate(-sample_ind[i])  # rotate the memory up to this index
            e = self.memory[0]  # sample from the top
            es.append(e.state)
            ea.append(e.action)
            er.append(e.reward)
            en.append(e.next_state)
            ed.append(e.done)
            self.memory.rotate(sample_ind[i])
            i += 1
        states = torch.stack(es).squeeze().float().to(device)
        actions = torch.stack(ea).float().to(device)
        rewards = torch.from_numpy(np.vstack(er)).float().to(device)
        next_states = torch.stack(en).squeeze().float().to(device)
        dones = torch.from_numpy(np.vstack(ed).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
 