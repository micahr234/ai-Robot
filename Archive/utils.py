import scipy.signal
import numpy as np

class waveform:

    def __init__(self, type='constant', period=1.0, min=0, max=1.0):
        self.type = type
        self.period = period
        self.min = min
        self.max = max

    def __call__(self, timestep):
        if self.type == 'triangle':
            out = (scipy.signal.sawtooth(2 * np.pi * (timestep-1) / self.period, width=0.5) * 0.5 + 0.5) * (self.max - self.min) + self.min
        elif self.type == 'constant':
            out = self.min
        elif self.type == 'sinusoid':
            out = (scipy.signal.sin(2 * np.pi * (timestep-1) / self.period) * 0.5 + 0.5) * (self.max - self.min) + self.min
        else:
            out = np.zeros_like(timestep)

        return out

    def __str__(self):
        if self.type == 'triangle':
            out = 'triangle wave - period: ' + str(self.period) + ' min: ' + str(self.min) + ' max: ' + str(self.max)
        elif self.type == 'constant':
            out = 'constant: ' + str(self.min)
        elif self.type == 'sinusoid':
            out = 'sinusoidal wave - period: ' + str(self.period) + ' min: ' + str(self.min) + ' max: ' + str(self.max)
        else:
            out = 'undefined'

        return out