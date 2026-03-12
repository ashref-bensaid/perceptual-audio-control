import numpy as np


def normalize(x):
    eps = 1e-8
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def smooth(x, alpha=0.9):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y


def build_descriptor_vector(centroid, bandwidth, flatness, alpha=0.9):
    c_norm = normalize(centroid)
    b_norm = normalize(bandwidth)
    f_norm = normalize(flatness)

    c_s = smooth(c_norm, alpha)
    b_s = smooth(b_norm, alpha)
    f_s = smooth(f_norm, alpha)

    # Shape: (frames, 3)
    return np.vstack([c_s, b_s, f_s]).T