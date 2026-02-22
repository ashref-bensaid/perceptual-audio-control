import numpy as np


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)


def exponential_smoothing(x, alpha=0.1):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y