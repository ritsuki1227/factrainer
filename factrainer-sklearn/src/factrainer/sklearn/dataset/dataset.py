from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
from factrainer.base.dataset import IndexableDataset, RowIndex, RowsAndColumns
from pydantic import field_validator
from typing_extensions import TypeIs

from sklearn.model_selection._split import _BaseKFold

from .types import MatrixLike, PdDataFrameLike, PdSeriesLike, VectorLike


class SklearnDataset(IndexableDataset):
    X: MatrixLike
    y: VectorLike | None = None

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        for train_index, val_index in k_fold.split(self.X):
            yield train_index.tolist(), val_index.tolist()

    def __getitem__(self, index: RowsAndColumns) -> "SklearnDataset":
        X: MatrixLike
        match index:
            case int():
                if self.y is None:
                    y = None
                elif isinstance(self.y, np.ndarray):
                    y = (
                        np.expand_dims(self.y[index], axis=0)
                        if self.y.ndim == 1
                        else self.y[index]
                    )
                elif self._is_pd_series(self.y):
                    y = self.y.take([index])
                else:
                    raise ValueError
                if isinstance(self.X, np.ndarray):
                    X = np.expand_dims(self.X[index], axis=0)
                elif self._is_pd_dataframe(self.X):
                    X = self.X.take([index])
                else:
                    raise ValueError
                return SklearnDataset(
                    X=X,
                    y=y,
                )
            case list():
                if self.y is None:
                    y = None
                elif isinstance(self.y, np.ndarray):
                    y = self.y[index]
                elif self._is_pd_series(self.y):
                    y = self.y.take(index)
                else:
                    ValueError
                if isinstance(self.X, np.ndarray):
                    X = self.X[index]
                elif self._is_pd_dataframe(self.X):
                    X = self.X.take(index)
                else:
                    ValueError
                return SklearnDataset(X=X, y=y)
            case slice():
                return SklearnDataset(
                    X=self.X[index], y=self.y[index] if self.y is not None else None
                )
            case tuple():
                raise NotImplementedError
            case _:
                raise NotImplementedError

    @field_validator("X", mode="after")
    @classmethod
    def validate_X(cls, value: Any) -> MatrixLike:
        if isinstance(value, np.ndarray):
            return value
        if cls._is_pd_dataframe(value):
            return value
        raise ValueError

    @field_validator("y", mode="after")
    @classmethod
    def validate_y(cls, value: Any) -> VectorLike | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if cls._is_pd_series(value):
            return value
        raise ValueError

    @classmethod
    def _is_pd_dataframe(cls, obj: Any) -> TypeIs[PdDataFrameLike]:
        try:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)
        except ImportError:
            return False

    @classmethod
    def _is_pd_series(cls, obj: Any) -> TypeIs[PdSeriesLike]:
        try:
            import pandas as pd

            return isinstance(obj, pd.Series)
        except ImportError:
            return False
