from collections.abc import Sequence

from base.dataset import BaseDataset, DataIndex, IndexableDataset


class IndexedDataset[T: IndexableDataset](BaseDataset):
    index: DataIndex
    data: T

    def __len__(self) -> int:
        return len(self.index)


class IndexedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[IndexedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])
