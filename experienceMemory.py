import numpy as np
from pathlib import Path
import torch

class ExperienceMemory():
    def __init__(self, max_size, num_of_states, filename, weight_initialization=1, weight_exponent=1):
        self.count = 0
        self.max_size = max_size
        self.weight_exponent = weight_exponent
        self.memory_state = torch.empty([self.max_size, num_of_states], requires_grad=False)
        self.memory_action = torch.empty([self.max_size, 1], requires_grad=False)
        self.memory_reward = torch.empty([self.max_size, 1], requires_grad=False)
        self.memory_next_state = torch.empty([self.max_size, num_of_states], requires_grad=False)
        self.memory_done = torch.empty([self.max_size, 1], requires_grad=False)
        self.memory_weight = torch.ones([self.max_size], requires_grad=False) * weight_initialization

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

    def prepare_dataset(self, dataset_size, replacement):
        index = torch.multinomial(self.get_all_weights(), dataset_size, replacement).tolist() # introduces bias to stocastic environments
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

    def add(self, item, weight):
        state, action, reward, next_state, done = item

        if self.count >= self.max_size:
            raise ValueError('Memory buffer overflow.')

        index = self.count
        self.memory_state[index, :] = torch.tensor(state, requires_grad=False)
        self.memory_action[index, :] = torch.tensor(action, requires_grad=False)
        self.memory_reward[index, :] = torch.tensor(reward, requires_grad=False)
        self.memory_next_state[index, :] = torch.tensor(next_state, requires_grad=False)
        self.memory_done[index, :] = torch.tensor(done, requires_grad=False)
        self.memory_weight[index] = torch.tensor(weight, requires_grad=False)
        self.count += 1
        pass

    def save(self):
        index = range(self.count)
        state, action, reward, next_state, done = self.get(index)
        file = Path(self.filename)
        torch.save({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}, file)
        pass

    def get_weight(self, index):
        weight = self.memory_weight[index]
        return weight

    def get_all_weights(self):
        index = range(self.count)
        weights = self.get_weight(index)
        return weights

    def set_weight(self, index, weight, factor):
        self.memory_weight[index] = (torch.squeeze(weight, dim=-1)**self.weight_exponent) * factor + self.memory_weight[index] * (1.0-factor)
        pass