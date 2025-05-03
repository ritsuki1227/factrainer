import numpy as np
from factrainer.core.cv.dataset import (
    IndexedDataset,
    IndexedDatasets,
    SplittedDataset,
    SplittedDatasets,
    SplittedDatasetsIndices,
)
from factrainer.sklearn.dataset.dataset import SklearnDataset
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold


class TestSplittedDatasets:
    def test_create_from_k_fold(self) -> None:
        k_fold = KFold(n_splits=3, shuffle=True, random_state=1)
        dataset = SklearnDataset(
            X=np.array([[1, 2], [3, 4], [5, 6]]), y=np.array([100, 200, 300])
        )
        expected = SplittedDatasets[SklearnDataset](
            datasets=[
                SplittedDataset(
                    train=IndexedDataset(
                        index=[1, 2],
                        data=SklearnDataset(
                            X=np.array([[3, 4], [5, 6]]), y=np.array([200, 300])
                        ),
                    ),
                    val=IndexedDataset(
                        index=[0],
                        data=SklearnDataset(X=np.array([[1, 2]]), y=np.array([100])),
                    ),
                    test=IndexedDataset(
                        index=[0],
                        data=SklearnDataset(X=np.array([[1, 2]]), y=np.array([100])),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 1],
                        data=SklearnDataset(
                            X=np.array([[1, 2], [3, 4]]), y=np.array([100, 200])
                        ),
                    ),
                    val=IndexedDataset(
                        index=[2],
                        data=SklearnDataset(X=np.array([[5, 6]]), y=np.array([300])),
                    ),
                    test=IndexedDataset(
                        index=[2],
                        data=SklearnDataset(X=np.array([[5, 6]]), y=np.array([300])),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 2],
                        data=SklearnDataset(
                            X=np.array([[1, 2], [5, 6]]),
                            y=np.array([100, 300]),
                        ),
                    ),
                    val=IndexedDataset(
                        index=[1],
                        data=SklearnDataset(X=np.array([[3, 4]]), y=np.array([200])),
                    ),
                    test=IndexedDataset(
                        index=[1],
                        data=SklearnDataset(X=np.array([[3, 4]]), y=np.array([200])),
                    ),
                ),
            ],
        )

        actual = SplittedDatasets.create(dataset, k_fold)

        for actual_dataset, expected_dataset in zip(actual.datasets, expected.datasets):
            assert_array_equal(actual_dataset.train.index, expected_dataset.train.index)
            assert_array_equal(actual_dataset.val.index, expected_dataset.val.index)  # type: ignore
            assert_array_equal(actual_dataset.test.index, expected_dataset.test.index)
            assert_array_equal(
                actual_dataset.train.data.X,
                expected_dataset.train.data.X,
            )
            assert_array_equal(
                actual_dataset.train.data.y,
                expected_dataset.train.data.y,
            )
            assert_array_equal(
                actual_dataset.val.data.X,  # type: ignore
                expected_dataset.val.data.X,  # type: ignore
            )
            assert_array_equal(
                actual_dataset.val.data.y,  # type: ignore
                expected_dataset.val.data.y,  # type: ignore
            )
            assert_array_equal(
                actual_dataset.test.data.X,
                expected_dataset.test.data.X,
            )
            assert_array_equal(
                actual_dataset.test.data.y,
                expected_dataset.test.data.y,
            )

    def test_create_from_indices(self) -> None:
        k_fold = SplittedDatasetsIndices(
            train=[[1, 2], [0, 1], [0, 2]], val=None, test=[[0], [2], [1]]
        )
        dataset = SklearnDataset(
            X=np.array([[1, 2], [3, 4], [5, 6]]), y=np.array([100, 200, 300])
        )
        expected = SplittedDatasets[SklearnDataset](
            datasets=[
                SplittedDataset(
                    train=IndexedDataset(
                        index=[1, 2],
                        data=SklearnDataset(
                            X=np.array([[3, 4], [5, 6]]), y=np.array([200, 300])
                        ),
                    ),
                    val=None,
                    test=IndexedDataset(
                        index=[0],
                        data=SklearnDataset(X=np.array([[1, 2]]), y=np.array([100])),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 1],
                        data=SklearnDataset(
                            X=np.array([[1, 2], [3, 4]]), y=np.array([100, 200])
                        ),
                    ),
                    val=None,
                    test=IndexedDataset(
                        index=[2],
                        data=SklearnDataset(X=np.array([[5, 6]]), y=np.array([300])),
                    ),
                ),
                SplittedDataset(
                    train=IndexedDataset(
                        index=[0, 2],
                        data=SklearnDataset(
                            X=np.array([[1, 2], [5, 6]]),
                            y=np.array([100, 300]),
                        ),
                    ),
                    val=None,
                    test=IndexedDataset(
                        index=[1],
                        data=SklearnDataset(X=np.array([[3, 4]]), y=np.array([200])),
                    ),
                ),
            ],
        )
        actual = SplittedDatasets.create(dataset, k_fold)
        for actual_dataset, expected_dataset in zip(actual.datasets, expected.datasets):
            assert actual_dataset.val is None
            assert_array_equal(actual_dataset.train.index, expected_dataset.train.index)
            assert_array_equal(actual_dataset.test.index, expected_dataset.test.index)
            assert_array_equal(
                actual_dataset.train.data.X,
                expected_dataset.train.data.X,
            )
            assert_array_equal(
                actual_dataset.train.data.y,
                expected_dataset.train.data.y,
            )
            assert_array_equal(
                actual_dataset.test.data.X,
                expected_dataset.test.data.X,
            )


class TestIndexedDatasets:
    def test_create(self) -> None:
        dataset = SklearnDataset(
            X=np.array([[1, 2], [3, 4], [5, 6]]), y=np.array([100, 200, 300])
        )
        indices = [[0, 1], [1, 2], [2, 0]]
        expected = IndexedDatasets[SklearnDataset](
            datasets=[
                IndexedDataset(
                    index=[0, 1],
                    data=SklearnDataset(
                        X=np.array([[1, 2], [3, 4]]), y=np.array([100, 200])
                    ),
                ),
                IndexedDataset(
                    index=[1, 2],
                    data=SklearnDataset(
                        X=np.array([[3, 4], [5, 6]]), y=np.array([200, 300])
                    ),
                ),
                IndexedDataset(
                    index=[2, 0],
                    data=SklearnDataset(
                        X=np.array([[5, 6], [1, 2]]), y=np.array([300, 100])
                    ),
                ),
            ]
        )
        actual = IndexedDatasets.create(dataset, indices)
        for actual_dataset, expected_dataset in zip(actual.datasets, expected.datasets):
            assert_array_equal(actual_dataset.index, expected_dataset.index)
            assert_array_equal(
                actual_dataset.data.X,
                expected_dataset.data.X,
            )
            assert_array_equal(
                actual_dataset.data.y,
                expected_dataset.data.y,
            )
