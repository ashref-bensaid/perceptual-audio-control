import sounddevice as sd
import numpy as np

from src.processors.spectral_shaper import PerceptualSpectralShaper


class RealtimePerceptualProcessor:

    def __init__(self, sr=44100, block_size=2048):

        self.sr = sr
        self.block_size = block_size

        self.processor = PerceptualSpectralShaper()

    def audio_callback(self, indata, outdata, frames, time, status):

        audio_in = indata[:, 0]

        y_out, _, _, _ = self.processor.process(audio_in, self.sr)

        outdata[:, 0] = y_out[:frames]

    def start(self):

        with sd.Stream(
            samplerate=self.sr,
            blocksize=self.block_size,
            channels=1,
            callback=self.audio_callback
        ):
            print("Realtime processor running...")
            input("Press Enter to stop")