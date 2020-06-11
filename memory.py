from pathlib import Path
import torch

class Memory():

    def __init__(self, capacity, filename):
        self.position = 0
        self.length = 0
        self.capacity = capacity
        self.memory_state = None
        self.memory_action = None
        self.memory_reward = None
        self.memory_next_state = None
        self.memory_done = None

        self.filename = Path(filename)

        if self.filename.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + str(self.filename))
            buffer = torch.load(self.filename)
            self.create(buffer['state'].shape[1:], buffer['action'].shape[1:])
            self.position = min(self.capacity-1, buffer['position'])
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

    def create(self, state_shape, action_shape):
        self.memory_state = torch.empty([self.capacity] + list(state_shape), device='cpu', requires_grad=False)
        self.memory_action = torch.empty([self.capacity] + list(action_shape), device='cpu', requires_grad=False)
        self.memory_reward = torch.empty([self.capacity, 1], device='cpu', requires_grad=False)
        self.memory_next_state = torch.empty([self.capacity] + list(state_shape), device='cpu', requires_grad=False)
        self.memory_done = torch.empty([self.capacity, 1], device='cpu', requires_grad=False)

    def get(self, index):
        if self.length == 0:
            raise ValueError('Memory buffer empty')
        state = self.memory_state[index, :]
        action = self.memory_action[index, :]
        reward = self.memory_reward[index, :]
        next_state = self.memory_next_state[index, :]
        done = self.memory_done[index, :]
        return state, action, reward, next_state, done

    def sample(self, batch_size):
        index = torch.randint(0, len(self), (batch_size,))
        state, action, reward, next_state, done = self.get(index)
        return state, action, reward, next_state, done

    def __len__(self):
        return self.length

    def add(self, state, action, reward, next_state, done):
        if self.length == 0:
            self.create(state.shape[1:], action.shape[1:])
        index = self.position
        self.memory_state[index, :] = state
        self.memory_action[index, :] = action
        self.memory_reward[index, :] = reward
        self.memory_next_state[index, :] = next_state
        self.memory_done[index, :] = done
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)
        pass

    def save(self):
        if self.length == 0:
            raise ValueError('Memory buffer empty')
        index = range(self.length)
        state, action, reward, next_state, done = self.get(index)
        torch.save({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done, 'length': self.length, 'position': self.position}, self.filename)
        pass