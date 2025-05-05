from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generator, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection._split import _BaseKFold
from typing_extensions import TypeIs

# type Prediction = npt.NDArray[Any] | scipy.sparse.spmatrix | list[scipy.sparse.spmatrix]
type Prediction = npt.NDArray[np.number[Any]]
type RowIndex = list[int]
type RowIndices = Sequence[RowIndex]
type Rows = int | slice | RowIndex


class BaseDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndexableDataset(BaseDataset):
    @abstractmethod
    def __getitem__(self, index: Rows) -> Self:
        raise NotImplementedError

    @abstractmethod
    def k_fold_split(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[RowIndex, RowIndex], None, None]:
        raise NotImplementedError


class BaseDatasetSlicer[T](ABC):
    @abstractmethod
    def slice(self, data: T, index: RowIndex) -> T:
        raise NotImplementedError


class IsImportableInstance[T](ABC):
    @abstractmethod
    def is_instance(self, obj: Any) -> TypeIs[T]:
        raise NotImplementedError
