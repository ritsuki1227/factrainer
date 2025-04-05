from abc import ABC, abstractmethod

from factrainer.base.config import BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import BaseDataset, Prediction
from factrainer.base.raw_model import RawModel


class PredictorTrait[T: BaseDataset, U: RawModel, W: BasePredictConfig](ABC):
    @abstractmethod
    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_model(self) -> U:
        raise NotImplementedError

    @property
    @abstractmethod
    def pred_config(self) -> W | None:
        raise NotImplementedError

    @pred_config.setter
    @abstractmethod
    def pred_config(self, config: W | None) -> None:
        raise NotImplementedError


class TrainerTrait[T: BaseDataset, V: BaseTrainConfig](ABC):
    @abstractmethod
    def train(self, dataset: T) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_config(self) -> V:
        raise NotImplementedError

    @train_config.setter
    @abstractmethod
    def train_config(self, config: V) -> None:
        raise NotImplementedError


class ValidatableTrainerTrait[T: BaseDataset, V: BaseTrainConfig](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_config(self) -> V:
        raise NotImplementedError

    @train_config.setter
    @abstractmethod
    def train_config(self, config: V) -> None:
        raise NotImplementedError
