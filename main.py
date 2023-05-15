from copy import deepcopy

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sample_rate = 16000
device = 'cpu'
model_name = 'theodotus/stt_uk_squeezeformer_ctc_ml'
data_dir = 'words_dataset'
audio_extension = '.wav'
num_workers = 0
batch_size = 16

asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name, map_location=torch.device(device)).to(device)

train_ds = deepcopy(asr_model.cfg.train_ds)
validation_ds = deepcopy(asr_model.cfg.validation_ds)
test_ds = deepcopy(asr_model.cfg.test_ds)

train_ds.manifest_filepath = 'manifest/train_manifest.json'
train_ds.sample_rate = sample_rate
train_ds.batch_size = batch_size
train_ds.is_tarred = False
train_ds.num_workers = num_workers

validation_ds.manifest_filepath = 'manifest/val_manifest.json'
validation_ds.sample_rate = sample_rate
validation_ds.batch_size = batch_size
validation_ds.num_workers = num_workers

test_ds.manifest_filepath = 'manifest/test_manifest.json'
test_ds.sample_rate = sample_rate
test_ds.batch_size = batch_size
test_ds.num_workers = num_workers

asr_model.setup_training_data(train_ds)
asr_model.setup_multiple_validation_data(validation_ds)
asr_model.setup_multiple_test_data(test_ds)

from pytorch_lightning.callbacks import ModelCheckpoint

max_epochs = 30
tb_logger = TensorBoardLogger('logs/', name='my_experiment')
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    filename='model.ckpt',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    logger=tb_logger,
    accelerator='cpu',
    max_epochs=max_epochs,
    accumulate_grad_batches=batch_size,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stop_callback],
   # resume_from_checkpoint='model.ckpt'
)

asr_model.set_trainer(trainer)

asr_model._wer.use_cer = True
asr_model._wer.log_prediction = True

trainer.fit(asr_model)
trainer.save_checkpoint("model.ckpt")
asr_model.save_to("model.nemo")
