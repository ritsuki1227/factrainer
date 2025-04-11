from collections.abc import Generator

from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns
from sklearn.model_selection._split import _BaseKFold

import lightgbm as lgb

from .equality_checker import LgbDatasetEqualityChecker
from .slicer import LgbDatasetSlicer


class LgbDataset(IndexableDataset):
    dataset: lgb.Dataset

    def get_index(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        for train_index, val_index in k_fold.split(self.dataset.data):  # type: ignore
            yield train_index.tolist(), val_index.tolist()

    def __getitem__(self, index: RowsAndColumns) -> "LgbDataset":
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
            case tuple():
                raise NotImplementedError
            case _:
                raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LgbDataset):
            return False
        return LgbDatasetEqualityChecker().check(self.dataset, value.dataset)
