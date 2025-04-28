from __future__ import annotations

import numpy as np
from factrainer.sklearn.dataset import SklearnDataset
from numpy.testing import assert_array_equal


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
