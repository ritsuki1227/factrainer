import lightgbm as lgb
import numpy as np
from factrainer.core.cv.dataset import (
    IndexedDataset,
    SplittedDataset,
    SplittedDatasets,
    SplittedDatasetsIndices,
)
from factrainer.lightgbm.dataset.dataset import LgbDataset
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold


class TestSplittedDatasets:
    def test_create_from_k_fold(self) -> None:
        k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
        dataset = LgbDataset(dataset=lgb.Dataset(np.array([[1, 2], [3, 4], [5, 6]])))
        expected = SplittedDatasets[LgbDataset](
            datasets=[
                SplittedDataset(
                    train=IndexedDataset(
                        index=[1, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[3, 4], [5, 6]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[0],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[1, 2]]))),
                    ),
                    test=IndexedDataset(
                        index=[0],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[1, 2]]))),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 1],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[1, 2], [3, 4]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[2],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[5, 6]]))),
                    ),
                    test=IndexedDataset(
                        index=[2],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[5, 6]]))),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[1, 2], [5, 6]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[1],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[3, 4]]))),
                    ),
                    test=IndexedDataset(
                        index=[1],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[3, 4]]))),
                    ),
                ),
            ]
        )

        actual = SplittedDatasets[LgbDataset].create(dataset, k_fold)

        for actual_dataset, expected_dataset in zip(actual.datasets, expected.datasets):
            assert_array_equal(actual_dataset.train.index, expected_dataset.train.index)
            assert_array_equal(actual_dataset.val.index, expected_dataset.val.index)  # type: ignore
            assert_array_equal(actual_dataset.test.index, expected_dataset.test.index)

            assert_array_equal(
                actual_dataset.train.data.dataset.data,
                expected_dataset.train.data.dataset.data,
            )
            assert_array_equal(
                actual_dataset.val.data.dataset.data,  # type: ignore
                expected_dataset.val.data.dataset.data,  # type: ignore
            )
            assert_array_equal(
                actual_dataset.test.data.dataset.data,
                expected_dataset.test.data.dataset.data,
            )

    def test_create_from_indices(self) -> None:
        k_fold = SplittedDatasetsIndices(
            train=[[1, 2], [0, 1], [0, 2]], val=[[0], [2], [1]], test=[[0], [2], [1]]
        )
        dataset = LgbDataset(dataset=lgb.Dataset(np.array([[1, 2], [3, 4], [5, 6]])))
        expected = SplittedDatasets[LgbDataset](
            datasets=[
                SplittedDataset(
                    train=IndexedDataset(
                        index=[1, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[3, 4], [5, 6]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[0],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[1, 2]]))),
                    ),
                    test=IndexedDataset(
                        index=[0],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[1, 2]]))),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 1],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[1, 2], [3, 4]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[2],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[5, 6]]))),
                    ),
                    test=IndexedDataset(
                        index=[2],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[5, 6]]))),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 2],
                        data=LgbDataset(
                            dataset=lgb.Dataset(np.array([[1, 2], [5, 6]]))
                        ),
                    ),
                    val=IndexedDataset(
                        index=[1],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[3, 4]]))),
                    ),
                    test=IndexedDataset(
                        index=[1],
                        data=LgbDataset(dataset=lgb.Dataset(np.array([[3, 4]]))),
                    ),
                ),
            ]
        )

        actual = SplittedDatasets[LgbDataset].create(dataset, k_fold)

        for actual_dataset, expected_dataset in zip(actual.datasets, expected.datasets):
            assert_array_equal(actual_dataset.train.index, expected_dataset.train.index)
            assert_array_equal(actual_dataset.val.index, expected_dataset.val.index)  # type: ignore
            assert_array_equal(actual_dataset.test.index, expected_dataset.test.index)

            assert_array_equal(
                actual_dataset.train.data.dataset.data,
                expected_dataset.train.data.dataset.data,
            )
            assert_array_equal(
                actual_dataset.val.data.dataset.data,  # type: ignore
                expected_dataset.val.data.dataset.data,  # type: ignore
            )
            assert_array_equal(
                actual_dataset.test.data.dataset.data,
                expected_dataset.test.data.dataset.data,
            )
