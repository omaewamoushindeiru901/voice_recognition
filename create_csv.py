import json
import os
import random
import shutil

import librosa
import pandas as pd
from tqdm import tqdm

sample_rate = 16000
data_dir = 'dataset/'
audio_extension = '.wav'

# get a list of all the audio files and their labels
audio_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(audio_extension):
            audio_path = os.path.join(root + "/" + filename)
            label = os.path.basename(root)
            audio_files.append((audio_path, label))

df = pd.DataFrame(audio_files)
df.to_csv("transcription.csv", index=False)
