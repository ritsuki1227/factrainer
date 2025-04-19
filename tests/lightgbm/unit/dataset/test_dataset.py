from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from factrainer.lightgbm import LgbDataset
from pandas.testing import assert_frame_equal


class TestLgbDatasetGetitem:
    def test_single_row(self) -> None:
        sut = LgbDataset(dataset=lgb.Dataset(data=np.array([[1, 2, 3], [4, 5, 6]])))
        expected = LgbDataset(dataset=lgb.Dataset(data=np.array([[1, 2, 3]])))
        actual = sut[0]
        assert actual == expected

    def test_rows(self) -> None:
        sut = LgbDataset(
            dataset=lgb.Dataset(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        )
        expected = LgbDataset(
            dataset=lgb.Dataset(data=np.array([[7, 8, 9], [1, 2, 3]]))
        )
        actual = sut[[2, 0]]
        assert actual == expected

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
