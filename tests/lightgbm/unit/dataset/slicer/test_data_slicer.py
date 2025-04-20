import numpy as np
import pandas as pd
from factrainer.lightgbm.dataset.slicer import LgbDataSlicer
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


def test_numpy_array() -> None:
    data = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.array([[5, 6], [1, 2]])
    sut = LgbDataSlicer()
    actual = sut.slice(data, [2, 0])
    assert_array_equal(actual, expected)


def test_pandas_df() -> None:
    data = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    expected = pd.DataFrame([[5, 6], [1, 2]], index=[2, 0])
    sut = LgbDataSlicer()
    actual = sut.slice(data, [2, 0])
    assert_frame_equal(actual, expected)
