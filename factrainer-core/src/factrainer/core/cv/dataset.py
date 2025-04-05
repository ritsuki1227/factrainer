from collections.abc import Sequence
from typing import Self

from factrainer.base.dataset import (
    BaseDataset,
    DataIndex,
    DataIndices,
    IndexableDataset,
)
from sklearn.model_selection._split import _BaseKFold


class IndexedDataset[T: IndexableDataset](BaseDataset):
    index: DataIndex
    data: T

    def __len__(self) -> int:
        return len(self.index)


class IndexedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[IndexedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])

    @classmethod
    def create(cls, dataset: T, k_fold: _BaseKFold | DataIndices) -> Self:
        raise NotImplementedError

    @property
    def indices(self) -> DataIndices:
        return [dataset.index for dataset in self.datasets]


class SplittedDataset[T: IndexableDataset](BaseDataset):
    train: IndexedDataset[T]
    val: IndexedDataset[T] | None
    test: IndexedDataset[T]


class SplittedDatasetsIndices(BaseDataset):
    train: DataIndices
    val: DataIndices | None
    test: DataIndices


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
            train_dataset, val_dataset, test_dataset = dataset.split(
                train_index, val_index, test_index
            )
            datasets.append(
                SplittedDataset(
                    train=IndexedDataset(index=train_index, data=train_dataset),
                    val=IndexedDataset(index=val_index, data=val_dataset),
                    test=IndexedDataset(index=test_index, data=test_dataset),
                )
            )
        return cls(datasets=datasets)
