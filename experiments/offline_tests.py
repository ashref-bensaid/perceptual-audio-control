import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from src.perceptual.brightness import spectral_centroid,perceptual_brightness_control,framewise_spectral_centroid, exponential_smoothing, apply_spectral_tilt
from src.perceptual.plotting import plot_spectrogram_with_control
from src.perceptual.analysis import compute_centroid, compute_bandwidth, compute_flatness
from src.control.control import build_descriptor_vector
from src.control.mapping import build_parameter_vector
from src.dsp.multi_transform import apply_multiaxis_transform
from scipy.ndimage import gaussian_filter1d
from src.realtime.realtime_processor import RealtimePerceptualProcessor
import librosa
import soundfile as sf

from src.processors.spectral_shaper import PerceptualSpectralShaper

y, sr = librosa.load("audio/input/Snare (Factory).wav", sr=None)

processor = PerceptualSpectralShaper()
processor.set_mode("enhance")
processor.set_strength(0.8)

y_out, M, M_mod, Theta = processor.process(y, sr)

sf.write("audio/output/snare_shaped.wav", y_out, sr)


plt.figure(figsize=(12,6))


plt.plot(Theta[:,0], label="Brightness α(t)")
plt.plot(Theta[:,1], label="Bandwidth γ(t)")
plt.plot(Theta[:,2], label="Texture η(t)")

plt.legend()
plt.title("Perceptual Control Parameters Over Time")
plt.xlabel("Frame")
plt.ylabel("Parameter Value")

plt.show()

plt.subplot(2,1,1)
librosa.display.specshow(
    librosa.amplitude_to_db(M, ref=np.max),
    sr=sr,
    hop_length=512,
    y_axis="log",
    x_axis="time"
)
plt.title("Original Spectrogram")

plt.subplot(2,1,2)
librosa.display.specshow(
    librosa.amplitude_to_db(M_mod, ref=np.max),
    sr=sr,
    hop_length=512,
    y_axis="log",
    x_axis="time"
)
plt.title("Spectral Shaper Output")

plt.tight_layout()
plt.show()



processor = RealtimePerceptualProcessor()

processor.start()
# y, sr = librosa.load("audio/input/Snare (Factory).wav", sr=None)
# M, phase = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))
# f = librosa.fft_frequencies(sr=sr, n_fft=2048)
#
# centroid = spectral_centroid(y, sr)
# print(f"Spectral Centroid: {centroid:.2f} Hz")
#
#
# y_out, centroid, control = perceptual_brightness_control(
#     y,
#     sr,
#     strength=1.0,
#     smoothing=0.1
# )
#
#
# # --- Compute frame-wise centroid ---
# centroid = framewise_spectral_centroid(y, sr)
#
# # --- Normalize & smooth ---
# control = exponential_smoothing((centroid - np.min(centroid)) / (np.max(centroid)-np.min(centroid)), alpha=0.1)
#
# # --- Apply spectral tilt ---
# y_out, modified_mag = apply_spectral_tilt(y, sr, control, strength=1.0)
#
#
# sf.write("audio/output/bright_control.wav", y_out, sr)
#
# plt.figure()
# plt.plot(centroid)
# plt.title("Spectral Centroid")
# plt.show()
#
# plt.figure()
# plt.plot(control)
# plt.title("Smoothed Control")
# plt.show()
#
#
# # --- Existing processing ---
# y_out, modified_mag = apply_spectral_tilt(y, sr, control, strength=1.0)
#
# # --- Plot ---
# plot_spectrogram_with_control(y, y_out, sr, control, n_fft=2048, hop_length=512)
#
#
# centroid = compute_centroid(y, sr)
# bandwidth = compute_bandwidth(y, sr)
# flatness = compute_flatness(y)
# D = build_descriptor_vector(centroid, bandwidth, flatness, alpha=0.9)
# Theta = build_parameter_vector(D)
# Theta[:,0] = gaussian_filter1d(Theta[:,0], sigma=3)
# Theta[:,1] = gaussian_filter1d(Theta[:,1], sigma=3)
# Theta[:,2] = gaussian_filter1d(Theta[:,2], sigma=3)
# # #phase 2
# # plt.figure(figsize=(10, 6))
# # plt.plot(D[:, 0], label="Brightness (Centroid)")
# # plt.plot(D[:, 1], label="Bandwidth (Spread)")
# # plt.plot(D[:, 2], label="Flatness (Noisiness)")
# # plt.legend()
# # plt.title("Smoothed Perceptual Descriptor Vector")
# # plt.xlabel("Frame")
# # plt.ylabel("Normalized Value")
# # plt.show()
# #
# # #phase 3B
# # plt.figure(figsize=(10, 6))
# # plt.plot(Theta[:, 0], label="Tilt α(t)")
# # plt.plot(Theta[:, 1], label="Concentration γ(t)")
# # plt.plot(Theta[:, 2], label="Smoothing η(t)")
# # plt.legend()
# # plt.title("Perceptual Control Parameters")
# # plt.xlabel("Frame")
# # plt.ylabel("Parameter Value")
# # plt.show()
#
#
#
#
# #phase 3 C
# # 4. Apply multi-axis transform
# M_mod = apply_multiaxis_transform(M, f, Theta)
# # 5. Reconstruct audio
# y_mod = librosa.istft(M_mod * np.exp(1j*np.angle(librosa.stft(y, n_fft=2048, hop_length=512))), hop_length=512)
# sf.write("audio/output/Snare_transformed.wav", y_mod, sr)
#
#
#
# plt.figure(figsize=(12,6))
#
# plt.subplot(2,1,1)
# librosa.display.specshow(librosa.amplitude_to_db(M, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
# plt.title("Original Spectrogram")
#
# plt.subplot(2,1,2)
# librosa.display.specshow(librosa.amplitude_to_db(M_mod, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
# plt.title("Transformed Spectrogram")
#
# print("Original mean magnitude:", np.mean(M))
# print("Modified mean magnitude:", np.mean(M_mod))
#
# plt.tight_layout()
# plt.show()