from abc import ABC, abstractmethod

from base.dataset import Dataset, Prediction
from base.raw_model import RawModel


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
    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        raise NotImplementedError
