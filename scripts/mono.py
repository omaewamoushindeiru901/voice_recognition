import os
import glob
from pydub import AudioSegment


def convert_to_mono(file_path):
    sound = AudioSegment.from_wav(file_path)
    sound = sound.set_channels(1)
    return sound


def process_directory(root_dir):
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(dirpath, file)
                print("Processing file:", file_path)
                mono_sound = convert_to_mono(file_path)
                output_file = file_path[:-4] + ".wav"
                mono_sound.export(output_file, format="wav")
                print("Converted to mono and saved to:", output_file)


root_dir = "../data"
process_directory(root_dir)
