from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import scipy
from numpy import typing as npt

from lightgbm.compat import (  # type: ignore
    dt_DataTable,
    pa_Array,
    pa_ChunkedArray,
)

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pandas._typing import Axis


class PdDataFrameProtocol(Protocol):
    def take(
        self, indices: list[int], axis: Axis = ..., **kwargs: Any
    ) -> "PdDataFrameProtocol": ...


class PdSeriesProtocol[T](Protocol):
    def take(
        self, indices: list[int], axis: Axis = ..., **kwargs: Any
    ) -> "PdSeriesProtocol[T]": ...


class PaTableProtocol(Protocol): ...


if TYPE_CHECKING:
    type pd_DataFrame = pd.DataFrame
    type pd_Series[T] = pd.Series[T]
    type pa_Table = pa.Table
else:
    type pd_DataFrame = PdDataFrameProtocol
    type pd_Series = PdSeriesProtocol
    type pa_Table = PaTableProtocol

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
    pa_Array[Any],
    pa_ChunkedArray[Any],
)

LgbWeightType = TypeVar(
    "LgbWeightType",
    list[float],
    list[int],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pa_Array[Any],
    pa_ChunkedArray[Any],
)

LgbInitScoreType = TypeVar(
    "LgbInitScoreType",
    list[float],
    list[list[float]],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pd_DataFrame,
    pa_Table,
    pa_Array[Any],
    pa_ChunkedArray[Any],
)

LgbGroupType = TypeVar(
    "LgbGroupType",
    list[float],
    list[int],
    npt.NDArray[Any],
    "pd_Series[Any]",
    pa_Array[Any],
    pa_ChunkedArray[Any],
)

LgbPositionType = TypeVar(
    "LgbPositionType",
    npt.NDArray[Any],
    "pd_Series[Any]",
)
