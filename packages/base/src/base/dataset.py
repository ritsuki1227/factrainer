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


class BaseDatasetEqualityChecker[T](ABC):
    @abstractmethod
    def check(self, left: T, right: T) -> bool:
        raise NotImplementedError


class BaseDatasetSlicer[T](ABC):
    @abstractmethod
    def slice(self, data: T, index: DataIndex) -> T:
        raise NotImplementedError
