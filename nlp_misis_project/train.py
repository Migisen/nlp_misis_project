from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from nlp_misis_project.datasets.sarc import SARCDataModule
from nlp_misis_project.models.any_bert import AnyBertClassifier

torch.set_float32_matmul_precision('medium')

# Конфиг
BATCH_SIZE = 8
ENABLE_LOGGER = True
WORDS_CUTOFF = 320
LR = 2e-5
BASE_MODEL = 'microsoft/deberta-v3-base'

EARLY_STOPPING = EarlyStopping(monitor="val_epoch_acc", mode="max", patience=5)
CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_epoch_acc", mode="max", filename='result_model', save_top_k=1,
                                      verbose=True)

LOGGER = WandbLogger(log_model="all", project='nlp-roberta') if ENABLE_LOGGER else None

# Инициализация
trainer = Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=1,
    callbacks=[EARLY_STOPPING, CHECKPOINT_CALLBACK],
    logger=LOGGER,
    check_val_every_n_epoch=1,
)

model = AnyBertClassifier(lr=LR, base_model=BASE_MODEL)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.add_special_tokens({'additional_special_tokens': ['[PARENT]', '[COMMENT]']})
model.update_token_embedding_size(len(tokenizer))

data_module = SARCDataModule(
    path_to_data=Path(__file__).parent.parent / 'dataset' / 'sarc_09-12.csv',
    batch_size=BATCH_SIZE,
    words_cutoff=WORDS_CUTOFF,
    base_model=BASE_MODEL,
    tokenizer=tokenizer
)

# Обучение
trainer.fit(model, datamodule=data_module)

test_model = AnyBertClassifier.load_from_checkpoint(CHECKPOINT_CALLBACK.best_model_path,
                                                    token_embedding_size=len(tokenizer),
                                                    base_model=BASE_MODEL)
trainer.test(test_model, datamodule=data_module)
