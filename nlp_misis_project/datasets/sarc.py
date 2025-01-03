import logging
from pathlib import Path

import numpy as np
import polars
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SARCDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx][0]], dtype=torch.float), self.data[idx][1].squeeze(axis=0), \
            self.data[idx][
                2].squeeze(axis=0)


class SARCDataModule(pl.LightningDataModule):
    def __init__(self,
                 path_to_data: Path,
                 base_model: str = 'roberta-base',
                 processed_data_dir: Path | None = None,
                 file_delimiter: str = '\t',
                 words_cutoff: int = 35, batch_size: int = 32, num_workers: int = 15
                 ):
        super().__init__()
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
        self._processed_file_name = f'sarc_processed_{words_cutoff}_{model_prefix}'

        self._processed_file_path = self._processed_data_dir / self._processed_file_name
        self._file_delimiter = file_delimiter

    def prepare_data(self):
        # Проверяем, есть ли уже обработанные данные
        if self._processed_file_path.with_suffix('.npy').exists():
            logger.info(f'Обработанные данные уже существуют в {self._processed_file_path}')
            return

        logger.info('Подготовленные данные не найдены. Начинаем обработку...')

        # Обрабатываем данные
        scanned_csv = polars.scan_csv(
            self._path_to_data,
            separator=self._file_delimiter,
            has_header=False,
            with_column_names=lambda cols: [
                'label', 'comment', 'user', 'subreddit', 'score', 'up', 'down', 'date', 'timestamp', 'parent_comment',
                'embed_1', 'embed_2'
            ]
        )

        # Даунсэмплим
        min_class = scanned_csv.with_columns('label').group_by('label').len().min().select('len').collect().item()
        balanced_df = (scanned_csv
                       .select(['label', 'comment'])
                       .group_by('label')
                       .agg(polars.all().head(min_class))
                       .explode('comment')
                       )

        balanced_df = balanced_df.with_columns(
            polars.col('comment').map_elements(
                lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=self.words_cutoff,
                                         return_tensors='np'), return_dtype=polars.Object).alias('input'),
            polars.col('label')
        ).with_columns(
            polars.col('comment'),
            polars.col('label'),
            polars.col('input').map_elements(lambda x: x['input_ids'], return_dtype=polars.Object).alias('input_ids'),
            polars.col('input').map_elements(lambda x: x['attention_mask'], return_dtype=polars.Object).alias(
                'attention_mask'),
        ).select('label', 'input_ids', 'attention_mask')

        # Сохраняем обработанные данные
        result_df = balanced_df.collect().to_numpy()
        np.save(self._processed_file_path, result_df)
        logger.info(f'Обработанные данные сохранены в {self._processed_file_path}')

    def setup(self, stage: str):
        full_array = np.load(self._processed_file_path.with_suffix('.npy'), allow_pickle=True)
        train_val_array, test_array = train_test_split(full_array, test_size=0.1, random_state=42,
                                                       stratify=full_array[:, 0])
        if stage == 'fit':
            train_split, val_split = train_test_split(train_val_array, test_size=0.2, random_state=42,
                                                      stratify=train_val_array[:, 0])
            self.train_dataset = SARCDataset(train_split)
            self.val_dataset = SARCDataset(val_split)
        elif stage == 'test':
            self.test_dataset = SARCDataset(test_array)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    test_module = SARCDataModule(path_to_data=Path(__file__).parent.parent.parent / 'dataset' / 'sarc_09-12.csv')
    test_module.prepare_data()
