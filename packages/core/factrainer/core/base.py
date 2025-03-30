from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel, ConfigDict

import numpy as np
import numpy.typing as npt
from sklearn.model_selection._split import _BaseKFold

# type Prediction = npt.NDArray[Any] | scipy.sparse.spmatrix | list[scipy.sparse.spmatrix]
type Prediction = npt.NDArray[np.number[Any]]
type DataIndex = list[int]
type DataIndices = Sequence[DataIndex]


class RawModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseDataset(BaseModel): ...


class Dataset(BaseDataset):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainAndValDataset[T: Dataset](BaseDataset):
    train: T
    val: T


type ValidatableDataset[T: Dataset] = T | TrainAndValDataset[T]
