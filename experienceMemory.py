import numpy as np
from pathlib import Path
import torch
from collections import namedtuple
import random

class ExperienceMemory():

    def __init__(self, max_size, num_of_states, num_of_actions, filename):
        self.count = 0
        self.max_size = max_size
        self.memory_state = torch.empty([self.max_size, num_of_states], requires_grad=False)
        self.memory_action = torch.empty([self.max_size, num_of_actions], requires_grad=False)
        self.memory_reward = torch.empty([self.max_size, 1], requires_grad=False)
        self.memory_next_state = torch.empty([self.max_size, num_of_states], requires_grad=False)
        self.memory_done = torch.empty([self.max_size, 1], requires_grad=False)

        self.filename = filename
        file = Path(self.filename)

        if file.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + str(self.filename))
            buffer = torch.load(self.filename)

            temp = buffer['state']
            temp_index = temp.shape[0]
            if temp_index > self.max_size:
                temp_index = self.max_size

            self.count = temp_index
            self.memory_state[0:temp_index, :] = buffer['state'][0:temp_index, :]
            self.memory_action[0:temp_index, :] = buffer['action'][0:temp_index, :]
            self.memory_reward[0:temp_index, :] = buffer['reward'][0:temp_index, :]
            self.memory_next_state[0:temp_index, :] = buffer['next_state'][0:temp_index, :]
            self.memory_done[0:temp_index, :] = buffer['done'][0:temp_index, :]

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

    def prepare_dataset(self):
        index = torch.randperm(self.count)
        return index

    def get_batch(self, dataset_index, batch_size, batch_num):
        batch_start = (batch_num-1)*batch_size
        batch_end = batch_num*batch_size
        dataset_length = len(dataset_index)
        if batch_end >= dataset_length:
            batch_end = dataset_length
            last_batch = True
        else:
            last_batch = False
        batch_index = dataset_index[batch_start:batch_end]

        state, action, reward, next_state, done = self.get(batch_index)

        return batch_index, state, action, reward, next_state, done, last_batch

    def len(self):
        return self.count

    def add(self, state, action, reward, next_state, done):

        if self.count >= self.max_size:
            reduce = self.count - self.max_size + 1
            self.memory_state[:-reduce, :] = self.memory_state[reduce:, :].clone()
            self.memory_action[:-reduce, :] = self.memory_action[reduce:, :].clone()
            self.memory_reward[:-reduce, :] = self.memory_reward[reduce:, :].clone()
            self.memory_next_state[:-reduce, :] = self.memory_next_state[reduce:, :].clone()
            self.memory_done[:-reduce, :] = self.memory_done[reduce:, :].clone()
            self.count -= reduce

        index = self.count
        self.memory_state[index, :] = torch.tensor(state, requires_grad=False)
        self.memory_action[index, :] = torch.tensor(action, requires_grad=False)
        self.memory_reward[index, :] = torch.tensor(reward, requires_grad=False)
        self.memory_next_state[index, :] = torch.tensor(next_state, requires_grad=False)
        self.memory_done[index, :] = torch.tensor(done, requires_grad=False)
        self.count += 1
        pass

    def save(self):
        index = range(self.count)
        state, action, reward, next_state, done = self.get(index)
        file = Path(self.filename)
        torch.save({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}, file)
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