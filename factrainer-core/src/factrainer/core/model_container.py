from abc import ABC, abstractmethod
from typing import Any

from factrainer.base.config import BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import BaseDataset, Prediction
from factrainer.base.raw_model import RawModel


class BaseModelContainer[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](ABC):
    @abstractmethod
    def train(self, train_dataset: T, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, pred_dataset: T, *args: Any, **kwargs: Any) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_model(self) -> U:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_config(self) -> V:
        raise NotImplementedError

    @train_config.setter
    @abstractmethod
    def train_config(self, config: V) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def pred_config(self) -> W | None:
        raise NotImplementedError

    @pred_config.setter
    @abstractmethod
    def pred_config(self, config: W | None) -> None:
        raise NotImplementedError
