from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from sklearn.model_selection import KFold  # type: ignore

from factrainer.domain.cv import IndexedDataset, SplittedDataset, SplittedDatasets
from factrainer.lightgbm.dataset.dataset import (
    LgbDataset,
)

type Data = npt.NDArray[np.number[Any]] | pd.DataFrame | pd.Series[Any]


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            LgbDataset(
                dataset=lgb.Dataset(
                    data=np.array([[100, 10], [200, 20]]),
                    label=np.array([1000, 2000]),
                    reference=lgb.Dataset(data=np.array([[1000]])),
                )
            ),
            LgbDataset(
                dataset=lgb.Dataset(
                    data=np.array([[100, 10], [200, 20]]),
                    label=np.array([1000, 2000]),
                    reference=lgb.Dataset(data=np.array([[1000]])),
                )
            ),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            LgbDataset(
                dataset=lgb.Dataset(
                    data=pd.DataFrame(np.array([[100, 10], [200, 20]])),
                    label=pd.Series([1000, 2000]),
                )
            ),
            LgbDataset(
                dataset=lgb.Dataset(
                    data=pd.DataFrame(np.array([[100, 10], [200, 20]])),
                    label=pd.Series([1000, 2000]),
                )
            ),
            True,
            id="normal/pandas",
        ),
        pytest.param(
            LgbDataset(dataset=lgb.Dataset(data=np.array([[100, 10], [200, 20]]))),
            LgbDataset(
                dataset=lgb.Dataset(
                    data=pd.DataFrame(np.array([[100, 10], [200, 20]])),
                )
            ),
            False,
            id="negative/same-data-different-type",
        ),
        pytest.param(
            LgbDataset(dataset=lgb.Dataset(data=np.array([[100, 10], [200, 20]]))),
            None,
            False,
            id="negative/None",
        ),
    ],
)
def test_dataset_equality(left: Data, right: Data | None, expected: bool) -> None:
    actual = left == right
    assert actual is expected


@pytest.mark.parametrize(
    ["data"],
    [
        pytest.param(
            np.array([[100, 10], [200, 20], [300, 30], [400, 40]]), id="numpy"
        ),
        pytest.param(
            pd.DataFrame(np.array([[100, 10], [200, 20], [300, 30], [400, 40]])),
            id="pandas",
        ),
    ],
)
def test_get_indices(data: Data) -> None:
    k_fold = KFold(n_splits=2, shuffle=True, random_state=1)
    k_fold.split = MagicMock(
        return_value=iter(
            [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
        )
    )
    sut = LgbDataset(
        dataset=lgb.Dataset(
            data=data,
        )
    )
    expected = [([0, 1], [2, 3]), ([2, 3], [0, 1])]

    actual = sut.get_indices(k_fold=k_fold)

    assert list(actual) == expected
    k_fold.split.assert_called_once_with(sut.dataset.data)


def test_split() -> None:
    data = np.array([[100, 10], [200, 20], [300, 30], [400, 40]])
    label = np.array([1000, 2000, 3000, 4000])
    train_index, val_index, test_index = [0, 1], [3, 2], [3, 2]

    sut = LgbDataset(dataset=lgb.Dataset(data=data, label=label))
    expected_train = LgbDataset(
        dataset=lgb.Dataset(
            data=np.array([[100, 10], [200, 20]]), label=np.array([1000, 2000])
        )
    )
    expected_val = LgbDataset(
        dataset=lgb.Dataset(
            data=np.array([[400, 40], [300, 30]]),
            label=np.array([4000, 3000]),
            reference=expected_train.dataset,
        )
    )
    expected_test = LgbDataset(
        dataset=lgb.Dataset(
            data=np.array([[400, 40], [300, 30]]),
            label=np.array([4000, 3000]),
            reference=expected_train.dataset,
        )
    )

    actual = sut.split(train_index, val_index, test_index)

    assert (expected_train, expected_val, expected_test) == actual


class TestCreateSplittedDatasets:
    def test_share_holdouts(self) -> None:
        data = np.array([[100, 10], [200, 20], [300, 30], [400, 40]])
        k_fold = KFold(n_splits=2, shuffle=True, random_state=1)
        k_fold.split = MagicMock(
            return_value=iter(
                [
                    (np.array([0, 1]), np.array([3, 2])),
                    (np.array([3, 2]), np.array([0, 1])),
                ]
            )
        )
        dataset = LgbDataset(dataset=lgb.Dataset(data=data))
        expected = SplittedDatasets(
            datasets=[
                SplittedDataset(
                    train=IndexedDataset(
                        indices=[0, 1],
                        data=LgbDataset(
                            dataset=lgb.Dataset(data=np.array([[100, 10], [200, 20]]))
                        ),
                    ),
                    val=IndexedDataset(
                        indices=[3, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(
                                data=np.array([[400, 40], [300, 30]]),
                                reference=lgb.Dataset(
                                    data=np.array([[100, 10], [200, 20]])
                                ),
                            ),
                        ),
                    ),
                    test=IndexedDataset(
                        indices=[3, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(
                                data=np.array([[400, 40], [300, 30]]),
                                reference=lgb.Dataset(
                                    data=np.array([[100, 10], [200, 20]])
                                ),
                            ),
                        ),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        indices=[3, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(data=np.array([[400, 40], [300, 30]]))
                        ),
                    ),
                    val=IndexedDataset(
                        indices=[0, 1],
                        data=LgbDataset(
                            dataset=lgb.Dataset(
                                data=np.array([[100, 10], [200, 20]]),
                                reference=lgb.Dataset(
                                    data=np.array([[400, 40], [300, 30]])
                                ),
                            )
                        ),
                    ),
                    test=IndexedDataset(
                        indices=[0, 1],
                        data=LgbDataset(
                            dataset=lgb.Dataset(
                                data=np.array([[100, 10], [200, 20]]),
                                reference=lgb.Dataset(
                                    data=np.array([[400, 40], [300, 30]])
                                ),
                            )
                        ),
                    ),
                ),
            ]
        )

        actual = SplittedDatasets.create(k_fold=k_fold, dataset=dataset)

        assert actual == expected
