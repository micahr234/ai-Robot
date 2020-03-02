import numpy as np
from pathlib import Path
import torch
from collections import namedtuple
import random

class ExperienceMemory():

    def __init__(self, capacity, filename, num_of_states, num_of_actions):
        self.position = 0
        self.length = 0
        self.capacity = capacity
        self.memory_state = torch.empty([self.capacity, num_of_states], requires_grad=False)
        self.memory_action = torch.empty([self.capacity, num_of_actions], requires_grad=False)
        self.memory_reward = torch.empty([self.capacity, 1], requires_grad=False)
        self.memory_next_state = torch.empty([self.capacity, num_of_states], requires_grad=False)
        self.memory_done = torch.empty([self.capacity, 1], requires_grad=False)

        self.filename = Path(filename)

        if self.filename.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + str(self.filename))
            buffer = torch.load(self.filename)

            self.position = min(self.capacity, buffer['position'])
            self.length = min(self.capacity, buffer['length'])
            index = self.length
            self.memory_state[0:index, :] = buffer['state'][0:index, :]
            self.memory_action[0:index, :] = buffer['action'][0:index, :]
            self.memory_reward[0:index, :] = buffer['reward'][0:index, :]
            self.memory_next_state[0:index, :] = buffer['next_state'][0:index, :]
            self.memory_done[0:index, :] = buffer['done'][0:index, :]
        else:
            print('No experience buffer to load')

        pass

    def get(self, index):
        state = self.memory_state[index, :]
        action = self.memory_action[index, :]
        reward = self.memory_reward[index, :]
        next_state = self.memory_next_state[index, :]
        done = self.memory_done[index, :]
        return state, action, reward, next_state, done

    def sample(self, batch_size):
        index = torch.randint(0, self.length, (batch_size,))
        state, action, reward, next_state, done = self.get(index)
        return state, action, reward, next_state, done

    def __len__(self):
        return self.length

    def add(self, state, action, reward, next_state, done):
        index = self.position
        self.memory_state[index, :] = torch.tensor(state, dtype=torch.float32)
        self.memory_action[index, :] = torch.tensor(action, dtype=torch.float32)
        self.memory_reward[index, :] = torch.tensor(reward, dtype=torch.float32)
        self.memory_next_state[index, :] = torch.tensor(next_state, dtype=torch.float32)
        self.memory_done[index, :] = torch.tensor(done, dtype=torch.float32)
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)
        pass

    def save(self):
        index = range(self.length)
        state, action, reward, next_state, done = self.get(index)
        torch.save({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done, 'length': self.length, 'position': self.position}, self.filename)
        pass


Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceMemoryNew():

    def __init__(self, capacity, filename):

        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.filename = Path(filename)

        if self.filename.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + str(self.filename))
            buffer = torch.load(self.filename)
            self.memory = buffer['memory']
            del self.memory[capacity:]
        else:
            print('No experience buffer to load')

        pass

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        pass

    def save(self):
        torch.save({'memory': self.memory}, self.filename)
        pass