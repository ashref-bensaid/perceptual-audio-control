import numpy as np
import librosa

def spectral_centroid(y, sr):
    """
    Compute spectral centroid (brightness proxy).
    Returns centroid in Hz.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)
