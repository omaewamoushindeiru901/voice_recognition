import csv

import torch
import evaluate
import nemo.collections.asr as nemo_asr

# Config
batch_size = 16
device = 'cuda'  # or cpu
testset_file = 'transcription_S.csv'

# Load the test dataset
with open(testset_file, encoding='utf-8') as f:
    samples = list(csv.DictReader(f))

# Load the model
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("model.nemo")

# Disable decoding strategy for the model
asr_model.change_decoding_strategy(None)


# A util function to make batches
def make_batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# Temporary variables
predictions_all = []
references_all = []

# Inference in the batched mode
for batch in make_batches(samples, batch_size):
    paths = [it['path'] for it in batch]
    references = [it['text'] for it in batch]

    # Transcribe the audio files
    predictions = asr_model.transcribe(paths2audio_files=paths, batch_size=batch_size)

    # Remove ▁ symbols from predictions
    predictions = [prediction.replace('▁', ' ') for prediction in predictions]

    # Log outputs
    print('---')
    print('Predictions:')
    print(predictions)
    print('References:')
    print(references)
    print('---')

    # Add predictions and references
    predictions_all.extend(predictions)
    references_all.extend(references)

# Load evaluators
wer = evaluate.load('wer')
cer = evaluate.load('cer')

# Evaluate
wer_value = round(wer.compute(predictions=predictions_all, references=references_all), 4)
cer_value = round(cer.compute(predictions=predictions_all, references=references_all), 4)

# Print results
print('Final:')
print(f'WER: {wer_value} | CER: {cer_value}')

