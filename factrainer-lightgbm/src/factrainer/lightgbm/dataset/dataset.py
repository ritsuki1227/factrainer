from collections.abc import Generator

import numpy as np
from factrainer.base.dataset import IndexableDataset, RowIndex, Rows
from sklearn.model_selection._split import _BaseKFold

import lightgbm as lgb

from .slicer import LgbDatasetSlicer
from .types import IsPdDataFrame


class LgbDataset(IndexableDataset):
    dataset: lgb.Dataset

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        if isinstance(self.dataset.data, np.ndarray) or IsPdDataFrame().is_instance(
            self.dataset.data
        ):
            for train_index, val_index in k_fold.split(self.dataset.data):
                yield train_index.tolist(), val_index.tolist()
        else:
            raise NotImplementedError

    def __getitem__(self, index: Rows) -> "LgbDataset":
        match index:
            case int():
                return LgbDataset(
                    dataset=LgbDatasetSlicer(self.dataset.reference).slice(
                        self.dataset, [index]
                    )
                )
            case list():
                return LgbDataset(
                    dataset=LgbDatasetSlicer(self.dataset.reference).slice(
                        self.dataset, index
                    )
                )
            case slice():
                raise NotImplementedError
            case _:
                raise TypeError
