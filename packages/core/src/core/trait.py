from abc import ABC, abstractmethod

from base.dataset import BaseDataset, Prediction
from base.raw_model import RawModel


class PredictorTrait[T: BaseDataset, U: RawModel](ABC):
    @abstractmethod
    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_model(self) -> U:
        raise NotImplementedError


class TrainerTrait[T: BaseDataset](ABC):
    @abstractmethod
    def train(self, dataset: T) -> None:
        raise NotImplementedError


class ValidatableTrainerTrait[T: BaseDataset](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        raise NotImplementedError
