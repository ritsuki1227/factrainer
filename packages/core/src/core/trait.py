from abc import ABC, abstractmethod

from base.config import Prediction
from base.dataset import BaseDataset, Dataset
from base.raw_model import RawModel


class TrainValDataset[T: Dataset](BaseDataset):
    train: T
    val: T | None = None


class PredictorTrait[T: Dataset, U: RawModel](ABC):
    @abstractmethod
    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> U:
        raise NotImplementedError


class TrainerTrait[T: Dataset](ABC):
    @abstractmethod
    def train(self, dataset: T) -> None:
        raise NotImplementedError


class ValidatableTrainerTrait[T: Dataset](ABC):
    @abstractmethod
    def train(self, dataset: TrainValDataset[T]) -> None:
        raise NotImplementedError
