from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import polars as pl
from factrainer.sklearn.dataset.dataset import SklearnDataset
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from polars.testing import assert_series_equal as pl_assert_series_equal


class TestSklearnDatasetGetitem:
    def test_single_row(self) -> None:
        sut = SklearnDataset(X=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([10, 20]))
        expected = SklearnDataset(X=np.array([[4, 5, 6]]), y=np.array([20]))
        actual = sut[1]
        assert_array_equal(actual.X, expected.X)
        assert_array_equal(actual.y, expected.y)

    def test_rows(self) -> None:
        sut = SklearnDataset(
            X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), y=np.array([10, 20, 30])
        )
        expected = SklearnDataset(
            X=np.array([[7, 8, 9], [1, 2, 3]]), y=np.array([30, 10])
        )
        actual = sut[[2, 0]]
        assert_array_equal(actual.X, expected.X)
        assert_array_equal(actual.y, expected.y)

    def test_pandas(self) -> None:
        X = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        y = pd.Series([10, 20, 30])
        sut = SklearnDataset(X=X, y=y)
        expected_X = pd.DataFrame(
            {
                "a": [3, 1],
                "b": [6, 4],
            },
            index=[2, 0],
        )
        expected_y = pd.Series([30, 10], index=[2, 0])
        actual = sut[[2, 0]]
        assert_frame_equal(cast(pd.DataFrame, actual.X), expected_X)
        if not isinstance(actual.y, pd.Series):
            raise TypeError
        assert_series_equal(actual.y, expected_y)

    def test_polars(self) -> None:
        X = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        y = pl.Series([10, 20, 30])
        sut = SklearnDataset(X=X, y=y)
        expected_X = pl.DataFrame(
            {
                "a": [3, 1],
                "b": [6, 4],
            }
        )
        expected_y = pl.Series([30, 10])
        actual = sut[[2, 0]]
        pl_assert_frame_equal(cast(pl.DataFrame, actual.X), expected_X)
        pl_assert_series_equal(cast(pl.Series, actual.y), expected_y)
