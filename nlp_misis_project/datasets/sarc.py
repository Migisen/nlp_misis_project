import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SARCDataset(Dataset):
    def __init__(self, data: list[torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[2][idx], self.data[0][idx], self.data[1][idx]


class SARCDataModule(pl.LightningDataModule):
    def __init__(self,
                 path_to_data: Path,
                 base_model: str = 'roberta-base',
                 processed_data_dir: Path | None = None,
                 file_delimiter: str = ',',
                 words_cutoff: int = 35, batch_size: int = 32, num_workers: int = 15,
                 tokenizer: AutoTokenizer | None = None
                 ):
        super().__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._path_to_data = path_to_data
        self._base_model = base_model

        if not processed_data_dir:
            self._processed_data_dir = path_to_data.parent
        else:
            self._processed_data_dir = processed_data_dir

        self.words_cutoff = words_cutoff
        if '/' in base_model:
            model_prefix = base_model.split('/')[1]
        else:
            model_prefix = base_model
        self._processed_filename_base = f'sarc_tokenized_{words_cutoff}_{model_prefix}'

        self._tokenized_path = self._processed_data_dir / 'tokenized'
        self._prepared_path = self._processed_data_dir / 'prepared'

        if not self._tokenized_path.exists():
            self._tokenized_path.mkdir(parents=True)

        self._file_delimiter = file_delimiter

    def prepared_data_exists(self) -> bool:
        for stage in ['train', 'test', 'val']:
            path_to_check = self._tokenized_path / f'{self._processed_filename_base}_{stage}.pt'
            if not path_to_check.exists():
                return False
        return True

    def prepare_data(self):
        # Проверяем, есть ли уже обработанные данные
        if self.prepared_data_exists():
            return

        logger.info('Подготовленные данные не найдены. Начинаем обработку...')

        for stage in ['train', 'test', 'val']:
            stage_df = pd.read_csv(self._prepared_path / f'prepared_{stage}.csv', sep=self._file_delimiter)
            comments = stage_df['comment'].values.tolist()
            tokenized_data = self.tokenizer(comments, padding='max_length', truncation=True,
                                            max_length=self.words_cutoff, return_tensors='pt')

            input_ids = tokenized_data['input_ids']
            attention_mask = tokenized_data['attention_mask']
            labels = stage_df['label'].values

            tokenized_data = [input_ids, attention_mask, torch.from_numpy(labels)]
            torch.save(tokenized_data, self._tokenized_path / f'{self._processed_filename_base}_{stage}.pt')

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = SARCDataset(
                torch.load(self._tokenized_path / f'{self._processed_filename_base}_train.pt')
            )
            self.val_dataset = SARCDataset(
                torch.load(self._tokenized_path / f'{self._processed_filename_base}_val.pt')
            )
        elif stage == 'test':
            self.test_dataset = SARCDataset(
                torch.load(self._tokenized_path / f'{self._processed_filename_base}_test.pt'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    test_module = SARCDataModule(path_to_data=Path(__file__).parent.parent.parent / 'dataset' / 'sarc_09-12.csv')
    test_module.prepare_data()
