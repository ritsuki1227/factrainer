from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generator, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection._split import _BaseKFold

# type Prediction = npt.NDArray[Any] | scipy.sparse.spmatrix | list[scipy.sparse.spmatrix]
type Prediction = npt.NDArray[np.number[Any]]
type DataIndex = list[int]
type DataIndices = Sequence[DataIndex]


class RawModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Dataset(BaseDataset): ...


class IndexableDataset(Dataset):
    @abstractmethod
    def get_index(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[DataIndex, DataIndex], None, None]:
        raise NotImplementedError

    @abstractmethod
    def split(
        self, train_index: DataIndex, val_index: DataIndex, test_index: DataIndex
    ) -> tuple[Self, Self, Self]:
        raise NotImplementedError


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


class BaseDatasetEqualityChecker[T](ABC):
    @abstractmethod
    def check(self, left: T, right: T) -> bool:
        raise NotImplementedError


class BaseDatasetSlicer[T](ABC):
    @abstractmethod
    def slice(self, data: T, index: DataIndex) -> T:
        raise NotImplementedError


############


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
