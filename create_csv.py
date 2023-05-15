import os

import pandas as pd

sample_rate = 16000
data_dir = 'words_dataset/'
audio_extension = '.wav'

audio_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(audio_extension):
            audio_path = os.path.join(root + "/" + filename)
            label = os.path.basename(root)
            audio_files.append((audio_path, label))

df = pd.DataFrame(audio_files)
df.to_csv("transcription.csv", index=False)
