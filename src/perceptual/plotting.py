import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def plot_spectrogram_with_control(y_orig, y_processed, sr, control, n_fft=2048, hop_length=512, title="Spectral Tilt"):
    """
    Plot original and processed spectrograms with control curve overlay.

    Parameters
    ----------
    y_orig : np.ndarray
        Original audio
    y_processed : np.ndarray
        Processed audio
    sr : int
        Sample rate
    control : np.ndarray
        Control curve (0-1)
    """
    # --- Compute spectrograms ---
    S_orig = np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length))
    S_proc = np.abs(librosa.stft(y_processed, n_fft=n_fft, hop_length=hop_length))

    times = librosa.frames_to_time(np.arange(S_orig.shape[1]), sr=sr, hop_length=hop_length)

    # --- Create figure ---
    plt.figure(figsize=(12, 8))

    # Original
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_orig, ref=np.max),
                             sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.title("Original Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    # Overlay control curve
    plt.twinx()
    plt.plot(times, control, color='cyan', linewidth=2, label="Control Curve")
    plt.ylabel("Control (0-1)")
    plt.legend(loc="upper right")

    # Processed
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_proc, ref=np.max),
                             sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.title("Processed Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()