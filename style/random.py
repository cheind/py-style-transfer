import numpy as np

def white_noise(shape, sigma=1e-2):
    return np.random.randn(*shape).astype(np.float32)*sigma