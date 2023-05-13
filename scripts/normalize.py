import os
from pydub import AudioSegment


def get_audio_files(folder):
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_file_volume(filename):
    audio = AudioSegment.from_file(filename)
    return audio.dBFS


def get_average_volume(audio_files):
    total_volume = 0
    for file in audio_files:
        total_volume += get_file_volume(file)
    return total_volume / len(audio_files)


def normalize_audio_volume(audio_file, target_volume):
    audio = AudioSegment.from_file(audio_file, format="wav")
    volume_diff = target_volume - get_file_volume(audio_file)
    return audio + volume_diff


audio_files = get_audio_files("dataset")
average_volume = get_average_volume(audio_files)

for file in audio_files:
    audio = normalize_audio_volume(file, average_volume)
    output_file = os.path.join(os.path.dirname(file), 'normalized_' + os.path.basename(file))
    print(output_file)
    audio.export(output_file, format='wav')
