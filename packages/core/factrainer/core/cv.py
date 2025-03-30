from collections.abc import Sequence

from .base import Dataset, Prediction, PredictorTrait, RawModel, TrainerTrait


class CvRawModels[U: RawModel](RawModel):
    models: Sequence[U]


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
