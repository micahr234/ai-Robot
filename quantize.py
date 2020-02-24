import numpy as np

def quantize(value, min, max, points):
    q = np.arange(points) / (points-1) * (max-min) + min
    idx = (np.abs(q - value)).argmin()
    return idx

def unquantize(idx, min, max, points):
    q = np.arange(points) / (points-1) * (max-min) + min
    value = q[idx]
    return value