from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel, ConfigDict

from base.dataset import Dataset, Prediction
from base.raw_model import RawModel


class BaseTrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BasePredictConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseLearner[T: Dataset, U: RawModel, V: BaseTrainConfig](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None, config: V) -> U:
        raise NotImplementedError


class BasePredictor[T: Dataset, U: RawModel, W: BasePredictConfig](ABC):
    @abstractmethod
    def predict(self, dataset: T, model: U, config: W | None) -> Prediction:
        raise NotImplementedError


class BaseMlModelConfig[
    T: Dataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](BaseModel, ABC):
    train_config: V
    learner: BaseLearner[T, U, V]
    predictor: BasePredictor[T, U, W]
    pred_config: W | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelConfigFactoryTrait[
    T: Dataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](ABC):
    @classmethod
    @abstractmethod
    def create(cls, train_config: V, pred_config: W | None = None) -> Self:
        raise NotImplementedError
