from collections.abc import Generator
from typing import Any

from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns
from numpy import typing as npt

from sklearn.model_selection._split import _BaseKFold


class SklearnDataset(IndexableDataset):
    X: npt.NDArray[Any]
    y: npt.NDArray[Any]

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        for train_index, val_index in k_fold.split(self.X):
            yield train_index.tolist(), val_index.tolist()

    def __getitem__(self, index: RowsAndColumns) -> "SklearnDataset":
        match index:
            case int():
                return SklearnDataset(X=self.X[index], y=self.y[index])
            case list():
                return SklearnDataset(X=self.X[index], y=self.y[index])
            case slice():
                return SklearnDataset(X=self.X[index], y=self.y[index])
            case tuple():
                raise NotImplementedError
            case _:
                raise NotImplementedError
