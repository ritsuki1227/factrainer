import lightgbm as lgb
import numpy as np
from factrainer.lightgbm import LgbDataset


class TestLgbDatasetGetitem:
    def test_row(self) -> None:
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
