import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from src.perceptual.brightness import spectral_centroid,perceptual_brightness_control,framewise_spectral_centroid, exponential_smoothing, apply_spectral_tilt
from src.perceptual.plotting import plot_spectrogram_with_control
y, sr = librosa.load("audio/input/Ducks.wav", sr=None)

centroid = spectral_centroid(y, sr)
print(f"Spectral Centroid: {centroid:.2f} Hz")


y_out, centroid, control = perceptual_brightness_control(
    y,
    sr,
    strength=1.0,
    smoothing=0.1
)


# --- Compute frame-wise centroid ---
centroid = framewise_spectral_centroid(y, sr)

# --- Normalize & smooth ---
control = exponential_smoothing((centroid - np.min(centroid)) / (np.max(centroid)-np.min(centroid)), alpha=0.1)

# --- Apply spectral tilt ---
y_out, modified_mag = apply_spectral_tilt(y, sr, control, strength=1.0)


sf.write("audio/output/bright_control.wav", y_out, sr)

plt.figure()
plt.plot(centroid)
plt.title("Spectral Centroid")
plt.show()

plt.figure()
plt.plot(control)
plt.title("Smoothed Control")
plt.show()




# --- Existing processing ---
y_out, modified_mag = apply_spectral_tilt(y, sr, control, strength=1.0)

# --- Plot ---
plot_spectrogram_with_control(y, y_out, sr, control, n_fft=2048, hop_length=512)