from collections.abc import Sequence
from typing import Self

from factrainer.base.dataset import (
    BaseDataset,
    IndexableDataset,
    RowIndex,
    RowIndices,
)
from sklearn.model_selection._split import _BaseKFold


class IndexedDataset[T: IndexableDataset](BaseDataset):
    index: RowIndex
    data: T

    def __len__(self) -> int:
        return len(self.index)


class IndexedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[IndexedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])

    @classmethod
    def create(cls, dataset: T, k_fold: _BaseKFold | RowIndices) -> Self:
        if isinstance(k_fold, _BaseKFold):
            raise NotImplementedError
        return cls(
            datasets=[
                IndexedDataset(index=index, data=dataset[index]) for index in k_fold
            ]
        )

    @property
    def indices(self) -> RowIndices:
        return [dataset.index for dataset in self.datasets]


class SplittedDataset[T: IndexableDataset](BaseDataset):
    train: IndexedDataset[T]
    val: IndexedDataset[T] | None
    test: IndexedDataset[T]


class SplittedDatasetsIndices(BaseDataset):
    train: RowIndices
    val: RowIndices | None
    test: RowIndices


class SplittedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[SplittedDataset[T]]

    @property
    def train(self) -> IndexedDatasets[T]:
        return IndexedDatasets(datasets=[dataset.train for dataset in self.datasets])

    @property
    def val(self) -> IndexedDatasets[T] | None:
        vals = []
        for dataset in self.datasets:
            if dataset.val is None:
                return None
            vals.append(dataset.val)
        return IndexedDatasets(datasets=vals)

    @property
    def test(self) -> IndexedDatasets[T]:
        return IndexedDatasets(datasets=[dataset.test for dataset in self.datasets])

    @property
    def indices(self) -> SplittedDatasetsIndices:
        return SplittedDatasetsIndices(
            train=self.train.indices,
            val=self.val.indices if self.val is not None else None,
            test=self.test.indices,
        )

    @classmethod
    def create(
        cls, dataset: T, k_fold: _BaseKFold, share_holdouts: bool = True
    ) -> Self:
        datasets = []
        for train_index, val_index in dataset.get_index(k_fold):
            if share_holdouts:
                test_index = val_index
            else:
                raise NotImplementedError
            datasets.append(
                SplittedDataset(
                    train=IndexedDataset(index=train_index, data=dataset[train_index]),
                    val=IndexedDataset(index=val_index, data=dataset[val_index]),
                    test=IndexedDataset(index=test_index, data=dataset[test_index]),
                )
            )
        return cls(datasets=datasets)
