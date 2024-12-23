import logging
from pathlib import Path

import numpy as np
import polars
import pytorch_lightning as pl
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)


class SARCDataset(Dataset):
    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


class SARCDataModule(pl.LightningDataModule):
    def __init__(self, path_to_data: Path, base_model: str = 'roberta-base', processed_data_dir: Path | None = None,
                 file_delimiter: str = '\t',
                 words_cutoff: int = 150):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)

        self._path_to_data = path_to_data

        if not processed_data_dir:
            self._processed_data_dir = path_to_data.parent
        else:
            self._processed_data_dir = processed_data_dir

        self.words_cutoff = words_cutoff
        self._processed_file_name = f'sarc_processed_{words_cutoff}'

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
                lambda x: self.tokenizer(x, padding='max_length', truncation=True, return_tensors='np'),
                strategy='thread_local', return_dtype=polars.Object).alias('input'),
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


if __name__ == '__main__':
    test_module = SARCDataModule(path_to_data=Path(__file__).parent.parent.parent / 'dataset' / 'sarc_09-12.csv')
    test_module.prepare_data()
