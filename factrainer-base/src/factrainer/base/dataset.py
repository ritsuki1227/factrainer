from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generator, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection._split import _BaseKFold

# type Prediction = npt.NDArray[Any] | scipy.sparse.spmatrix | list[scipy.sparse.spmatrix]
type Prediction = npt.NDArray[np.number[Any]]
type RowIndex = list[int]
type RowIndices = Sequence[RowIndex]
type Rows = int | slice | RowIndex
type RowsAndColumns = Rows | tuple[Rows, ...]


class BaseDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndexableDataset(BaseDataset):
    @abstractmethod
    def __getitem__(self, index: RowsAndColumns) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get_index(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        raise NotImplementedError


class BaseDatasetEqualityChecker[T](ABC):
    @abstractmethod
    def check(self, left: T, right: T) -> bool:
        raise NotImplementedError


class BaseDatasetSlicer[T](ABC):
    @abstractmethod
    def slice(self, data: T, index: RowIndex) -> T:
        raise NotImplementedError
