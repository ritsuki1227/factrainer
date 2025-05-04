from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from factrainer.lightgbm.dataset.slicer import LgbInitScoreSlicer
from numpy.testing import assert_array_equal


def test_numpy_1d_array() -> None:
    data = np.array([1, 2, 3, 4, 5, 6])
    expected = np.array([3, 1])
    sut = LgbInitScoreSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(actual, expected)


def test_numpy_2d_array() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.array([[5, 6], [1, 2]])
    sut = LgbInitScoreSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(actual, expected)


def test_pandas_series() -> None:
    data = pd.Series([1, 2, 3, 4, 5, 6])
    expected = pd.Series([3, 1], index=[2, 0])
    sut = LgbInitScoreSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(cast("pd.Series[Any]", actual), expected)


def test_pandas_dataframe() -> None:
    data = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    expected = pd.DataFrame([[5, 6], [1, 2]], index=[2, 0])
    sut = LgbInitScoreSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(cast(pd.DataFrame, actual), expected)
