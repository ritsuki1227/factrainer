from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection._split import _BaseKFold

# type Prediction = npt.NDArray[Any] | scipy.sparse.spmatrix | list[scipy.sparse.spmatrix]
type Prediction = npt.NDArray[np.number[Any]]
type DataIndices = list[int]


class BaseTrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BasePredictConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndexableDataset(BaseDataset):
    @abstractmethod
    def get_indices(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[DataIndices, DataIndices], None, None]:
        raise NotImplementedError

    @abstractmethod
    def split(
        self, train_index: DataIndices, val_index: DataIndices, test_index: DataIndices
    ) -> tuple[Self, Self, Self]:
        raise NotImplementedError


class BaseDatasetEqualityChecker[T](ABC):
    @abstractmethod
    def check(self, left: T, right: T) -> bool:
        raise NotImplementedError


class BaseDatasetSlicer[T](ABC):
    @abstractmethod
    def slice(self, data: T, index: DataIndices) -> T:
        raise NotImplementedError


class RawModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseLearner[T: BaseDataset, U: RawModel, V: BaseTrainConfig](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None, config: V) -> U:
        raise NotImplementedError


class BasePredictor[T: BaseDataset, U: RawModel, W: BasePredictConfig](ABC):
    @abstractmethod
    def predict(self, dataset: T, model: U, config: W | None) -> Prediction:
        raise NotImplementedError


class BaseMlModelConfig[
    T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](BaseModel, ABC):
    train_config: V
    learner: BaseLearner[T, U, V]
    predictor: BasePredictor[T, U, W]
    pred_config: W | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PresettableTrait[
    T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](ABC):
    @classmethod
    @abstractmethod
    def create(cls, train_config: V, pred_config: W | None = None) -> Self:
        raise NotImplementedError


class BaseMlModel[T: BaseDataset, U: RawModel](ABC):
    @abstractmethod
    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> U:
        raise NotImplementedError
