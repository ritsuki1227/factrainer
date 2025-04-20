import numpy as np
import pandas as pd
from factrainer.lightgbm.dataset.slicer import LgbLabelSlicer
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal


def test_numpy_1d_array() -> None:
    data = np.array([1, 2, 3, 4, 5, 6])
    expected = np.array([3, 1])
    sut = LgbLabelSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(actual, expected)


def test_numpy_2d_array() -> None:
    data = np.array([[1], [2], [3], [4], [5], [6]])
    expected = np.array([[3], [1]])
    sut = LgbLabelSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(actual, expected)


def test_pandas_series() -> None:
    data = pd.Series([1, 2, 3, 4, 5, 6])
    expected = pd.Series([3, 1], index=[2, 0])
    sut = LgbLabelSlicer()
    actual = sut.slice(data, [2, 0])
    assert_series_equal(actual, expected)


def test_pandas_df() -> None:
    data = pd.DataFrame([[1], [2], [3], [4], [5], [6]])
    expected = pd.DataFrame([[3], [1]], index=[2, 0])
    sut = LgbLabelSlicer()
    actual = sut.slice(data, [2, 0])
    assert_frame_equal(actual, expected)
