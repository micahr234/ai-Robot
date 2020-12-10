from pathlib import Path
import torch

class memory():

    def __init__(self, capacity, filename, device):
        self.position = 0
        self.length = 0
        self.capacity = capacity
        self.memory = {}
        self.device = device

        self.filename = Path(filename)

        if self.filename.is_file():
            
            # Load experience buffer
            print('Loading experience buffer from file ' + str(self.filename))
            buffer = torch.load(self.filename)
            self.position = min(self.capacity-1, buffer['position'])
            self.length = min(self.capacity, buffer['length'])
            del buffer['position']
            del buffer['length']
            index = self.length

            self.create(**buffer)
            for key in buffer:
                self.memory[key][0:index, :] = buffer[key][0:index, :]

        else:
            
            print('No experience buffer to load')

        pass

    def create(self, **kwargs):
        for key, value in kwargs.items():
            self.memory[key] = torch.zeros([self.capacity] + list(value.shape), device=self.device, requires_grad=False)
        pass

    def get(self, index):
        if self.length == 0:
            raise ValueError('Memory buffer empty')
        kwargs = {}
        for key, value in self.memory.items():
            kwargs[key] = value[index, :]
        return kwargs

    def get_all(self):
        kwargs = {}
        for key, value in self.memory.items():
            kwargs[key] = value[:len(self), :]
        return kwargs

    def sample(self, batch_size):
        index = torch.randint(0, len(self), (batch_size,))
        kwargs = self.get(index)
        return kwargs

    def __len__(self):
        return self.length

    def add(self, **kwargs):
        if self.length == 0:
            self.create(**kwargs)
        index = self.position
        for key, value in kwargs.items():
            self.memory[key][index, :] = value
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.capacity, self.length + 1)
        pass

    def save(self):
        if self.length == 0:
            raise ValueError('Memory buffer empty')
        index = range(self.length)
        kwargs = self.get(index)
        kwargs['length'] = self.length
        kwargs['position'] = self.position
        torch.save(kwargs, self.filename)
        pass