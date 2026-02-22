import numpy as np
import librosa
from perceptual.analysis import compute_stft, spectral_centroid_from_magnitude
from perceptual.utils import normalize, exponential_smoothing


def spectral_centroid(y, sr):
    """
    Compute spectral centroid (brightness proxy).
    Returns centroid in Hz.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def framewise_spectral_centroid(y, sr, n_fft=2048, hop_length=512):
    """
    Compute frame-wise spectral centroid.
    Returns 1D array (one value per frame).
    """
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return centroid[0]

def perceptual_brightness_control(
        y,
        sr,
        n_fft=2048,
        hop_length=512,
        strength=1.0,
        smoothing=0.1
):
    """
    Applies time-varying spectral tilt based on frame-wise spectral centroid.
    Returns:
        y_out: processed signal
        centroid: raw centroid curve
        control: smoothed control curve
    """

    # --- STFT ---
    magnitude, phase = compute_stft(y, sr, n_fft, hop_length)

    # --- Descriptor ---
    centroid = spectral_centroid_from_magnitude(magnitude, sr, n_fft)

    # --- Normalize ---
    centroid_norm = normalize(centroid)

    # --- Smooth control ---
    control = exponential_smoothing(centroid_norm, alpha=smoothing)

    # --- Map to tilt exponent ---
    alpha = 0.5 + strength * control   # clamp range ~ [0.5, 1.5]

    # --- Build frequency weighting ---
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = freqs[:, np.newaxis]
    freqs_norm = freqs / np.max(freqs)

    W = freqs_norm ** alpha[np.newaxis, :]

    # --- Apply spectral modification ---
    modified_magnitude = magnitude * W

    X_modified = modified_magnitude * np.exp(1j * phase)

    # --- Reconstruct ---
    y_out = librosa.istft(X_modified, hop_length=hop_length)

    return y_out, centroid, control


def apply_spectral_tilt(y, sr, control, n_fft=2048, hop_length=512, strength=1.0):
    """
    Apply time-varying spectral tilt based on a smoothed control curve.

    Parameters
    ----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    control : np.ndarray
        Smoothed control curve (0–1)
    n_fft : int
        FFT size
    hop_length : int
        Hop length
    strength : float
        Amount of tilt

    Returns
    -------
    y_out : np.ndarray
        Processed audio
    modified_magnitude : np.ndarray
        STFT magnitude after tilt (for plotting/diagnostics)
    """

    # --- STFT ---
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(X)
    phase = np.angle(X)

    # --- Map control curve to tilt exponent ---
    # Clamp to reasonable range
    alpha = 0.5 + strength * control  # alpha ~ [0.5, 1.5]

    # Make alpha shape compatible: (1, frames)
    alpha = alpha[np.newaxis, :]

    # --- Frequency axis ---
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = freqs[:, np.newaxis]  # shape: (freq_bins, 1)
    freqs_norm = freqs / np.max(freqs)  # normalize 0–1

    # --- Build spectral weighting ---
    W = freqs_norm ** alpha  # shape: (freq_bins, frames)

    # --- Apply weighting ---
    modified_magnitude = magnitude * W

    # --- Inverse STFT ---
    X_modified = modified_magnitude * np.exp(1j * phase)
    y_out = librosa.istft(X_modified, hop_length=hop_length)

    return y_out, modified_magnitude