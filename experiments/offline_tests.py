import librosa
from src.perceptual.brightness import spectral_centroid

y, sr = librosa.load("audio/input/Ducks.wav", sr=None)

centroid = spectral_centroid(y, sr)
print(f"Spectral Centroid: {centroid:.2f} Hz")