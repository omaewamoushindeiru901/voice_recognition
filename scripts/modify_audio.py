import os

import librosa
import numpy as np
import soundfile as sf
from librosa.effects import time_stretch, pitch_shift, hpss, trim
from librosa.util import normalize
from scipy import signal


def apply_effects_to_folder(folder):
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            filepath = os.path.join(folder, file)
            y, sr = librosa.load(filepath, sr=16000)

            for effect in ['pitch_shift', 'white_noise', 'pink_noise', 'brown_noise', 'percussive', 'trim']:
                if effect == 'time_stretch':
                    new = time_stretch(y, rate=0.8)
                elif effect == 'pitch_shift':
                    new = pitch_shift(y, sr=sr, n_steps=-3)
                elif effect == 'white_noise':
                    new = y + 0.005 * np.random.randn(len(y))
                elif effect == 'pink_noise':
                    pink_noise = np.random.randn(len(y))
                    b, a = signal.butter(1, 0.1)
                    pink_noise = signal.filtfilt(b, a, pink_noise)
                    new = y + 0.01 * pink_noise
                elif effect == 'brown_noise':
                    brown_noise = np.cumsum(np.random.randn(len(y)) * (sr ** -(1 / 2)))
                    b, a = signal.butter(1, 0.1)
                    brown_noise = signal.filtfilt(b, a, brown_noise)
                    new = y + 0.05 * brown_noise
                elif effect == 'harmonic':
                    new, _ = hpss(y, kernel_size=15)
                elif effect == 'percussive':
                    _, new = hpss(y, kernel_size=15)
                elif effect == 'trim':
                    new, _ = trim(y)
                new_file = os.path.splitext(file)[0] + effect + '.wav'
                sf.write(os.path.join(folder, new_file), normalize(new), sr)


dataset_path = "/sentences_dataset\\"
folders = os.listdir(dataset_path)
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        apply_effects_to_folder(folder_path)
