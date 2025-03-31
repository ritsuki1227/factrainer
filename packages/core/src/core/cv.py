from collections.abc import Sequence

from base.dataset import BaseDataset, DataIndex, Dataset, IndexableDataset, Prediction
from base.raw_model import RawModel

from .trait import PredictorTrait, TrainerTrait


class CvRawModels[U: RawModel](RawModel):
    models: Sequence[U]


class IndexedDataset[T: IndexableDataset](BaseDataset):
    index: DataIndex
    data: T

    def __len__(self) -> int:
        return len(self.index)


class IndexedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[IndexedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])


class CvMlModel[T: Dataset, U: RawModel](TrainerTrait[T], PredictorTrait[T, U]):
    def __init__(self) -> None:
        pass

    def train(self, dataset: T) -> None:
        raise NotImplementedError

    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    def model(self) -> U:
        raise NotImplementedError
