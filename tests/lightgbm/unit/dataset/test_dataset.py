from __future__ import annotations

from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from factrainer.lightgbm import LgbDataset
from numpy import typing as npt
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal


class TestLgbDatasetGetitem:
    def test_single_row(self) -> None:
        sut = LgbDataset(dataset=lgb.Dataset(data=np.array([[1, 2, 3], [4, 5, 6]])))
        expected = np.array([[1, 2, 3]])
        actual = sut[0]
        assert_array_equal(cast(npt.NDArray[Any], actual.dataset.data), expected)

    def test_rows(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        )
        expected = np.array([[7, 8, 9], [1, 2, 3]])
        actual = sut[[2, 0]]
        assert_array_equal(cast(npt.NDArray[Any], actual.dataset.data), expected)

    def test_pandas(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(
                data=pd.DataFrame(
                    {
                        "a": [1, 2, 3],
                        "b": ["4", pd.NA, "6"],
                    }
                )
            )
        )
        expected = pd.DataFrame(
            {
                "a": [2, 1],
                "b": [pd.NA, "4"],
            },
            index=[1, 0],
        )

        actual = sut[[1, 0]]
        assert_frame_equal(cast(pd.DataFrame, actual.dataset.data), expected)

    def test_labels(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(
                data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                label=np.array([10, 20, 30]),
            )
        )
        expected_data = np.array([[7, 8, 9], [1, 2, 3]])
        expected_labels = np.array([30, 10])

        actual = sut[[2, 0]]

        assert_array_equal(cast(npt.NDArray[Any], actual.dataset.data), expected_data)
        assert_array_equal(
            cast(npt.NDArray[Any], actual.dataset.label), expected_labels
        )

    def test_labels_pandas(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(
                data=pd.DataFrame(
                    {
                        "a": [1, 2, 3],
                        "b": ["4", pd.NA, "6"],
                    }
                ),
                label=pd.Series([10, 20, 30]),
            )
        )
        expected_data = pd.DataFrame(
            {
                "a": [2, 1],
                "b": [pd.NA, "4"],
            },
            index=[1, 0],
        )
        expected_labels = pd.Series([20, 10], index=[1, 0])

        actual = sut[[1, 0]]

        assert_frame_equal(cast(pd.DataFrame, actual.dataset.data), expected_data)
        if not isinstance(actual.dataset.label, pd.Series):
            raise TypeError
        assert_series_equal(actual.dataset.label, expected_labels)

    def test_fixed_attributes(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(
                data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                feature_name=["a", "b", "c"],
                categorical_feature=["a"],
                params={"foo": "bar"},
                free_raw_data=False,
            )
        )
        expected_data = np.array([[7, 8, 9], [1, 2, 3]])

        actual = sut[[2, 0]]

        assert_array_equal(cast(npt.NDArray[Any], actual.dataset.data), expected_data)
        assert actual.dataset.feature_name == ["a", "b", "c"]
        assert actual.dataset.categorical_feature == ["a"]
        assert actual.dataset.params == {"foo": "bar"}
        assert actual.dataset.free_raw_data is False
