from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns
from numpy import typing as npt

from sklearn.model_selection._split import _BaseKFold

type MatrixLike = npt.NDArray[Any]
type VectorLike = npt.NDArray[Any]


class SklearnDataset(IndexableDataset):
    X: MatrixLike
    y: VectorLike | None = None

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        for train_index, val_index in k_fold.split(self.X):
            yield train_index.tolist(), val_index.tolist()

    def __getitem__(self, index: RowsAndColumns) -> "SklearnDataset":
        match index:
            case int():
                if self.y is None:
                    y = None
                else:
                    y = (
                        np.expand_dims(self.y[index], axis=0)
                        if self.y.ndim == 1
                        else self.y[index]
                    )
                return SklearnDataset(
                    X=np.expand_dims(self.X[index], axis=0),
                    y=y,
                )
            case list():
                return SklearnDataset(
                    X=self.X[index],
                    y=self.y[index] if self.y is not None else None,
                )
            case slice():
                return SklearnDataset(
                    X=self.X[index], y=self.y[index] if self.y is not None else None
                )
            case tuple():
                raise NotImplementedError
            case _:
                raise NotImplementedError
