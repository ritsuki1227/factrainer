from collections.abc import Generator

import lightgbm as lgb
from base.dataset import DataIndex, IndexableDataset
from sklearn.model_selection._split import _BaseKFold

from .equality_checker import LgbDatasetEqualityChecker
from .slicer import LgbDatasetSlicer


class LgbDataset(IndexableDataset):
    dataset: lgb.Dataset

    def get_index(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[DataIndex, DataIndex], None, None]:
        for train_index, val_index in k_fold.split(self.dataset.data):  # type: ignore
            yield train_index.tolist(), val_index.tolist()

    def split(
        self, train_index: DataIndex, val_index: DataIndex, test_index: DataIndex
    ) -> tuple["LgbDataset", "LgbDataset", "LgbDataset"]:
        train_dataset = LgbDatasetSlicer(reference=None).slice(
            self.dataset, train_index
        )
        val_dataset = LgbDatasetSlicer(reference=train_dataset).slice(
            self.dataset, val_index
        )
        test_dataset = LgbDatasetSlicer(reference=train_dataset).slice(
            self.dataset, val_index
        )
        return (
            LgbDataset(dataset=train_dataset),
            LgbDataset(dataset=val_dataset),
            LgbDataset(dataset=test_dataset),
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LgbDataset):
            return False
        return LgbDatasetEqualityChecker().check(self.dataset, value.dataset)
