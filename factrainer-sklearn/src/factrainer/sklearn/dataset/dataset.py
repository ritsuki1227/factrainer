from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import polars as pl
from factrainer.base.dataset import IndexableDataset, RowIndex, Rows
from pydantic import field_validator

from sklearn.model_selection._split import _BaseKFold

from .types import (
    IsPdDataFrame,
    IsPdSeries,
    IsPlDataFrame,
    IsPlSeries,
    MatrixLike,
    VectorLike,
)


class SklearnDataset(IndexableDataset):
    """Wrapper for scikit-learn compatible data providing factrainer-compatible interface.

    This class wraps feature matrix and target vector to provide a consistent
    interface for cross-validation and data manipulation within the factrainer
    framework. It supports multiple data formats including numpy arrays, pandas
    DataFrames/Series, and polars DataFrames/Series.
    This dataset is passed to the `train` and `predict` methods of
    `SingleModelContainer` and `CvModelContainer` when using scikit-learn models.

    Attributes
    ----------
    X : MatrixLike
        Feature matrix. Can be a numpy array, pandas DataFrame, or polars DataFrame.
    y : VectorLike | None
        Target vector. Can be a numpy array, pandas Series, polars Series, or None.

    Examples
    --------
    >>> import numpy as np
    >>> from factrainer.sklearn import SklearnDataset
    >>> # Create from numpy arrays
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> dataset = SklearnDataset(X=X, y=y)
    >>> # Create without target
    >>> dataset = SklearnDataset(X=X)
    >>> # Create from pandas
    >>> import pandas as pd
    >>> df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    >>> target = pd.Series(y, name="target")
    >>> dataset = SklearnDataset(X=df, y=target)
    """

    X: MatrixLike
    y: VectorLike | None = None

    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        for train_index, val_index in k_fold.split(self.X):
            yield train_index, val_index

    def __getitem__(self, index: Rows) -> "SklearnDataset":
        return SklearnDataset(
            X=self._getitem_X(self.X, index), y=self._getitem_y(self.y, index)
        )

    def _getitem_X(self, X: MatrixLike, index: Rows) -> MatrixLike:
        match index:
            case int():
                if isinstance(X, np.ndarray):
                    return np.expand_dims(X[index], axis=0)
                elif IsPdDataFrame().is_instance(X):
                    return X.take([index])
                elif IsPlDataFrame().is_instance(X):
                    raise NotImplementedError
                else:
                    raise ValueError
            case list():
                if isinstance(X, np.ndarray):
                    return X[index]
                elif IsPdDataFrame().is_instance(X):
                    return X.take(index)
                elif IsPlDataFrame().is_instance(X):
                    return (
                        pl.DataFrame({"__polars_row_index__": index})
                        .with_row_index(name="__factrainer_row_index__")
                        .join(
                            X.with_row_index(name="__polars_row_index__"),
                            on="__polars_row_index__",
                        )
                        .sort("__factrainer_row_index__")
                        .select(
                            pl.exclude(
                                ["__factrainer_row_index__", "__polars_row_index__"]
                            )
                        )
                    )
                else:
                    raise ValueError
            case np.ndarray():
                if isinstance(X, np.ndarray):
                    return X[index]
                elif IsPdDataFrame().is_instance(X):
                    return X.take(index)
                elif IsPlDataFrame().is_instance(X):
                    return (
                        pl.DataFrame({"__polars_row_index__": index})
                        .with_row_index(name="__factrainer_row_index__")
                        .join(
                            X.with_row_index(name="__polars_row_index__"),
                            on="__polars_row_index__",
                        )
                        .sort("__factrainer_row_index__")
                        .select(
                            pl.exclude(
                                ["__factrainer_row_index__", "__polars_row_index__"]
                            )
                        )
                    )
                else:
                    raise ValueError
            case slice():
                return X[index]
            case tuple():
                raise NotImplementedError
            case _:
                raise NotImplementedError

    def _getitem_y(self, y: VectorLike | None, index: Rows) -> VectorLike | None:
        if y is None:
            return None
        match index:
            case int():
                if isinstance(y, np.ndarray):
                    return np.expand_dims(y[index], axis=0) if y.ndim == 1 else y[index]
                elif IsPdSeries().is_instance(y):
                    return y.take([index])
                elif IsPlSeries().is_instance(y):
                    raise NotImplementedError
                else:
                    raise ValueError
            case list():
                if isinstance(y, np.ndarray):
                    return y[index]
                elif IsPdSeries().is_instance(y):
                    return y.take(index)
                elif IsPlSeries().is_instance(y):
                    return (
                        pl.DataFrame({"__polars_row_index__": index})
                        .with_row_index(name="__factrainer_row_index__")
                        .join(
                            pl.DataFrame(y).with_row_index(name="__polars_row_index__"),
                            on="__polars_row_index__",
                        )
                        .sort("__factrainer_row_index__")
                        .select(
                            pl.exclude(
                                ["__factrainer_row_index__", "__polars_row_index__"]
                            )
                        )
                        .to_series()
                    )
                else:
                    raise ValueError
            case np.ndarray():
                if isinstance(y, np.ndarray):
                    return y[index]
                elif IsPdSeries().is_instance(y):
                    return y.take(index)
                elif IsPlSeries().is_instance(y):
                    return (
                        pl.DataFrame({"__polars_row_index__": index})
                        .with_row_index(name="__factrainer_row_index__")
                        .join(
                            pl.DataFrame(y).with_row_index(name="__polars_row_index__"),
                            on="__polars_row_index__",
                        )
                        .sort("__factrainer_row_index__")
                        .select(
                            pl.exclude(
                                ["__factrainer_row_index__", "__polars_row_index__"]
                            )
                        )
                        .to_series()
                    )
                else:
                    raise ValueError
            case slice():
                return y[index]
            case _:
                raise TypeError

    @field_validator("X", mode="after")
    @classmethod
    def validate_X(cls, value: Any) -> MatrixLike:
        if isinstance(value, np.ndarray):
            return value
        if IsPdDataFrame().is_instance(value):
            return value
        if IsPlDataFrame().is_instance(value):
            return value
        raise ValueError(
            "X must be a numpy array, pandas DataFrame, or polars DataFrame"
        )

    @field_validator("y", mode="after")
    @classmethod
    def validate_y(cls, value: Any) -> VectorLike | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if IsPdSeries().is_instance(value):
            return value
        if IsPlSeries().is_instance(value):
            return value
        raise ValueError("y must be a numpy array, pandas Series, or polars Series")
