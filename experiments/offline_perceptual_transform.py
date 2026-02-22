from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

from src.perceptual.brightness import spectral_centroid
from src.dsp.filters import apply_lowpass

# -------------------------------------------------
# Project root (robust path handling)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
audio_path = PROJECT_ROOT / "audio" / "input" / "Ducks.wav"

# -------------------------------------------------
# Load audio
# -------------------------------------------------
y, sr = librosa.load(audio_path, sr=None)

# -------------------------------------------------
# Compute spectral centroid
# -------------------------------------------------
centroid = spectral_centroid(y, sr)
print(f"Spectral Centroid: {centroid:.2f} Hz")

# -------------------------------------------------
# Map centroid to cutoff frequency
# -------------------------------------------------
# Define perceptual bounds
min_centroid = 500
max_centroid = 5000

low_cutoff = 800
high_cutoff = 8000

cutoff = np.interp(
    centroid,
    [min_centroid, max_centroid],
    [low_cutoff, high_cutoff]
)

print(f"Mapped Cutoff Frequency: {cutoff:.2f} Hz")

# -------------------------------------------------
# Apply filter
# -------------------------------------------------
y_filtered = apply_lowpass(y, sr, cutoff)

# -------------------------------------------------
# Save output
# -------------------------------------------------
output_path = PROJECT_ROOT / "audio" / "output" / "Ducks_filtered.wav"
sf.write(output_path, y_filtered, sr)

print(f"Filtered audio saved to: {output_path}")

# -------------------------------------------------
# Plot before/after spectrum
# -------------------------------------------------
plt.figure(figsize=(10, 6))

Y = np.abs(np.fft.rfft(y))
Yf = np.abs(np.fft.rfft(y_filtered))
freqs = np.fft.rfftfreq(len(y), 1/sr)

plt.plot(freqs, Y, alpha=0.5, label="Original")
plt.plot(freqs, Yf, alpha=0.8, label="Filtered")

plt.xlim(0, 10000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Spectrum Before and After Perceptual Control")

plt.show()
