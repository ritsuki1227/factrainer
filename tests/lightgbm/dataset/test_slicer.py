from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from factrainer.lightgbm.dataset.slicer import (
    LgbDataSlicer,
    LgbLabelSlicer,
)


class TestDataSlicer:
    def test_np_array(self) -> None:
        data = np.array([[100, 10], [200, 20], [300, 30], [400, 40]])
        index = [3, 2]
        expected = np.array([[400, 40], [300, 30]])
        sut = LgbDataSlicer()

        actual = sut.slice(data, index)

        assert_array_equal(actual, expected)

    def test_pd_dataframe(self) -> None:
        data = pd.DataFrame(
            np.array([[100, 10], [200, 20], [300, 30], [400, 40]]),
            index=[0, 1, 2, 3],
        )
        index = [3, 2]
        expected = pd.DataFrame(np.array([[400, 40], [300, 30]]), index=[3, 2])
        sut = LgbDataSlicer()

        actual = sut.slice(data, index)

        assert_frame_equal(actual, expected)


class TestLabelSlicer:
    def test_np_array(self) -> None:
        label = np.array([10, 20, 30, 40])
        index = [3, 2]
        expected = np.array([40, 30])
        sut = LgbLabelSlicer()

        actual = sut.slice(label, index)

        assert_array_equal(actual, expected)

    def test_pd_series(self) -> None:
        label = pd.Series([10, 20, 30, 40], index=[0, 1, 2, 3])
        index = [3, 2]
        expected = pd.Series([40, 30], index=[3, 2])
        sut = LgbLabelSlicer()

        actual = sut.slice(label, index)

        assert_series_equal(actual, expected)
