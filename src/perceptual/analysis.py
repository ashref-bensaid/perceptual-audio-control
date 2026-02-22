import numpy as np
import librosa


def compute_stft(y, sr, n_fft=2048, hop_length=512):
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(X)
    phase = np.angle(X)
    return magnitude, phase


def spectral_centroid_from_magnitude(magnitude, sr, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = freqs[:, np.newaxis]
    numerator = np.sum(freqs * magnitude, axis=0)
    denominator = np.sum(magnitude, axis=0) + 1e-10
    return numerator / denominator