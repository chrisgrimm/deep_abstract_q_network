import numpy as np
import os


screen_file = './replay_buffer.npy'
screens = np.load(screen_file)

def get_batch(batch_size, max_n):
    indices = np.random.randint(0, screens.shape[0]-max_n, size=batch_size)
    offsets = np.random.randint(0, max_n, size=batch_size)
    start = indices
    end = indices + offsets
    s1 = screens[start, :, :]
    s2 = screens[end, :, :]
    n = offsets
    return s1, s2, n