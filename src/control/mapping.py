import numpy as np


def map_brightness(c_norm, alpha_min=-1.0, alpha_max=1.0):
    return alpha_min + c_norm * (alpha_max - alpha_min)


def map_bandwidth(b_norm, gamma_max=1.0):
    return b_norm * gamma_max


def map_flatness(f_norm):
    # Direct mapping for now
    return 0.6 * f_norm


def build_parameter_vector(D):
    """
    D shape: (frames, 3)
    Columns: [brightness, bandwidth, flatness]
    """
    c = D[:, 0]
    b = D[:, 1]
    f = D[:, 2]

    alpha = map_brightness(c)
    gamma = map_bandwidth(b)
    eta = map_flatness(f)

    return np.vstack([alpha, gamma, eta]).T