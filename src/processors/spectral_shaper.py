import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.perceptual.analysis import compute_centroid, compute_bandwidth, compute_flatness
from src.control.control import build_descriptor_vector
from src.control.mapping import build_parameter_vector
from src.dsp.multi_transform import apply_multiaxis_transform
from src.realtime.max_bridge import send_parameters

class PerceptualSpectralShaper:

    def __init__(self, n_fft=2048, hop_length=512, strength=1.0, mode="neutral"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.strength = strength
        self.mode = mode
        self.prev_theta = None
        self.user_brightness = 1.0
        self.user_bandwidth = 1.0
        self.user_texture = 1.0


    def set_strength(self, value):
        self.strength = value

    def set_mode(self, mode):
        self.mode = mode

    def set_brightness(self, value):
        self.user_brightness = value

    def set_bandwidth(self, value):
        self.user_bandwidth = value

    def set_texture(self, value):
        self.user_texture = value

    def process(self, y, sr):

        # STFT
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        M, phase = librosa.magphase(S)

        f = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        # perceptual descriptors
        centroid = compute_centroid(y, sr)
        bandwidth = compute_bandwidth(y, sr)
        flatness = compute_flatness(y)

        # descriptor vector
        D = build_descriptor_vector(centroid, bandwidth, flatness)

        # mapping
        Theta = build_parameter_vector(D)

        # smoothing
        Theta[:,0] = gaussian_filter1d(Theta[:,0], sigma=3)
        Theta[:,1] = gaussian_filter1d(Theta[:,1], sigma=3)
        Theta[:,2] = gaussian_filter1d(Theta[:,2], sigma=3)
        send_parameters(Theta[-1])
        # mode behaviour
        if self.mode == "enhance":
            Theta[:, 0] *= 1.4  # spectral tilt
            Theta[:, 1] *= 1.2  # bandwidth

        elif self.mode == "soften":
            Theta[:, 0] *= 0.7
            Theta[:, 1] *= 0.8

        elif self.mode == "texture":
            Theta[:, 1] *= 0.7
            Theta[:, 2] *= 1.4

        Theta[:, 0] *= self.user_brightness
        Theta[:, 1] *= self.user_bandwidth
        Theta[:, 2] *= self.user_texture

        if self.prev_theta is not None:
            Theta = 0.8 * self.prev_theta + 0.2 * Theta

        self.prev_theta = Theta
        # strength scaling
        #Theta *= self.strength

        # spectral transform
        M_effect = apply_multiaxis_transform(M, f, Theta)

        # mix original and processed spectrum
        M_mod = (1 - self.strength) * M + self.strength * M_effect

        # reconstruction
        y_out = librosa.istft(M_mod * phase, hop_length=self.hop_length)

        return y_out, M, M_mod, Theta

    def process_frame(self, frame, sr):

        S = librosa.stft(frame, n_fft=self.n_fft, hop_length=self.hop_length)
        M, phase = librosa.magphase(S)

        f = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        centroid = compute_centroid(frame, sr)
        bandwidth = compute_bandwidth(frame, sr)
        flatness = compute_flatness(frame)

        D = build_descriptor_vector(centroid, bandwidth, flatness)
        Theta = build_parameter_vector(D)

        M_mod = apply_multiaxis_transform(M, f, Theta)

        y_out = librosa.istft(M_mod * phase, hop_length=self.hop_length)

        return y_out