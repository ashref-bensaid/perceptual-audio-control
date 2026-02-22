import numpy as np
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, sr, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass(y, sr, cutoff, order=4):
    b, a = butter_lowpass(cutoff, sr, order)
    return lfilter(b, a, y)