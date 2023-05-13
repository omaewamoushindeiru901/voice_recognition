import json
import os
import random

import librosa
from tqdm import tqdm

sample_rate = 16000
data_dir = 'data/'
audio_extension = '.wav'

train_manifest_path = "train_manifest.json"
val_manifest_path = "val_manifest.json"
test_manifest_path = "test_manifest.json"

# get a list of all the audio files and their labels
audio_files = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(audio_extension):
            audio_path = os.path.join(root + filename)
            label = filename.replace(audio_extension, "").replace('pitch_shift', "").replace('white_noise', "").replace('pink_noise', "").replace('percussive', "").replace('trim', "").replace('brown_noise', "").lower()
            audio_files.append((audio_path, label))

# shuffle the audio files randomly
random.shuffle(audio_files)

# split the data into train, validation, and test sets
train_size = int(0.8 * len(audio_files))
val_size = int(0.1 * len(audio_files))
test_size = len(audio_files) - train_size - val_size

train_data = audio_files[:train_size]
val_data = audio_files[train_size:train_size + val_size]
test_data = audio_files[train_size + val_size:]
# Create empty manifest files
with open(train_manifest_path, "w", encoding='utf-8') as f:
    json.dump([], f)
with open(val_manifest_path, "w", encoding='utf-8') as f:
    json.dump([], f)
with open(test_manifest_path, "w", encoding='utf-8') as f:
    json.dump([], f)

# Iterate through each audio file and append to corresponding manifest file
for data, manifest_path in [(train_data, train_manifest_path), (val_data, val_manifest_path),
                            (test_data, test_manifest_path)]:
    manifest = []
    for audio_file, label in tqdm(data):
        y, sr = librosa.load(audio_file, sr=sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        manifest.append({
            "audio_filepath": audio_file,
            "text": label,
            "duration": duration
        })
    with open(manifest_path, "w", encoding='utf-8') as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
