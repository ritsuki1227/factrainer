from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel, ConfigDict

from .dataset import BaseDataset, Prediction
from .raw_model import RawModel


class BaseTrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BasePredictConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseLearner[T: BaseDataset, U: RawModel, V: BaseTrainConfig](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None, config: V) -> U:
        raise NotImplementedError


class BasePredictor[T: BaseDataset, U: RawModel, W: BasePredictConfig](ABC):
    @abstractmethod
    def predict(self, dataset: T, raw_model: U, config: W | None) -> Prediction:
        raise NotImplementedError


class BaseMlModelConfig[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModel):
    learner: BaseLearner[T, U, V]
    predictor: BasePredictor[T, U, W]
    train_config: V
    pred_config: W | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MlModelConfig[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseMlModelConfig[T, U, V, W], ABC):
    @classmethod
    @abstractmethod
    def create(cls, train_config: V, pred_config: W | None = None) -> Self:
        raise NotImplementedError
