import numpy as np

def quantize(value, min, max, points):
    value = np.array(value)
    max = np.array(max)
    min = np.array(min)
    points = np.array(points)
    scaled_value = (value - min) / (max - min) * (points - 1)
    idx = scaled_value.round().astype(int)
    return idx

def unquantize(idx, min, max, points):
    idx = np.array(idx)
    max = np.array(max)
    min = np.array(min)
    points = np.array(points)
    value = idx / (points-1) * (max - min) + min
    return value