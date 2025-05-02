from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import scipy
from numpy import typing as npt
from typing_extensions import TypeIs

from lightgbm.compat import (
    dt_DataTable,
)

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pandas._typing import Axis


@runtime_checkable
class PdDataFrameProtocol(Protocol):
    def take(
        self, indices: list[int], axis: Axis = ..., **kwargs: Any
    ) -> "PdDataFrameProtocol": ...


@runtime_checkable
class PdSeriesProtocol[T](Protocol):
    def take(
        self, indices: list[int], axis: Axis = ..., **kwargs: Any
    ) -> "PdSeriesProtocol[T]": ...


@runtime_checkable
class PaTableProtocol(Protocol): ...


@runtime_checkable
class PaArrayProtocol(Protocol): ...


@runtime_checkable
class PaChunkedArrayProtocol(Protocol): ...


if TYPE_CHECKING:
    type pd_DataFrame = pd.DataFrame
    type pd_Series[T] = pd.Series[T]
    type pa_Table = pa.Table
    type pa_Array = pa.Array[Any]
    type pa_ChunkedArray = pa.ChunkedArray[Any]
else:
    type pd_DataFrame = PdDataFrameProtocol
    type pd_Series[T] = PdSeriesProtocol[T]
    type pa_Table = PaTableProtocol
    type pa_Array = PaArrayProtocol
    type pa_ChunkedArray = PaChunkedArrayProtocol

type LgbParams = dict[str, Any] | None

LgbDataType = TypeVar(
    "LgbDataType",
    str,
    Path,
    npt.NDArray[Any],
    pd_DataFrame,
    dt_DataTable,
    scipy.sparse.spmatrix,
    Sequence[Any],
    list[Sequence[Any]],
    list[npt.NDArray[Any]],
    pa_Table,
)

LgbLabelType = TypeVar(
    "LgbLabelType",
    list[float],
    list[int],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pd_DataFrame,
    pa_Array,
    pa_ChunkedArray,
)

LgbWeightType = TypeVar(
    "LgbWeightType",
    list[float],
    list[int],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pa_Array,
    pa_ChunkedArray,
)

LgbInitScoreType = TypeVar(
    "LgbInitScoreType",
    list[float],
    list[list[float]],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pd_DataFrame,
    pa_Table,
    pa_Array,
    pa_ChunkedArray,
)

LgbGroupType = TypeVar(
    "LgbGroupType",
    list[float],
    list[int],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pa_Array,
    pa_ChunkedArray,
)

LgbPositionType = TypeVar(
    "LgbPositionType",
    npt.NDArray[Any],
    "pd_Series[Any]",
)


class IsImportableInstance[T](ABC):
    @abstractmethod
    def is_instance(self, obj: Any) -> TypeIs[T]:
        raise NotImplementedError


class IsPdDataFrame(IsImportableInstance[pd_DataFrame]):
    def is_instance(self, obj: Any) -> TypeIs[pd_DataFrame]:
        try:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)
        except ImportError:
            return False


class IsPdSeries(IsImportableInstance[pd_Series[Any]]):
    def is_instance(self, obj: Any) -> TypeIs[pd_Series[Any]]:
        try:
            import pandas as pd

            return isinstance(obj, pd.Series)
        except ImportError:
            return False


class IsPaTable(IsImportableInstance[pa_Table]):
    def is_instance(self, obj: Any) -> TypeIs[pa_Table]:
        try:
            import pyarrow as pa

            return isinstance(obj, pa.Table)
        except ImportError:
            return False


class IsPaArray(IsImportableInstance[pa_Array]):
    def is_instance(self, obj: Any) -> TypeIs[pa_Array]:
        try:
            import pyarrow as pa

            return isinstance(obj, pa.Array)
        except ImportError:
            return False


class IsPaChunkedArray(IsImportableInstance[pa_ChunkedArray]):
    def is_instance(self, obj: Any) -> TypeIs[pa_ChunkedArray]:
        try:
            import pyarrow as pa

            return isinstance(obj, pa.ChunkedArray)
        except ImportError:
            return False
