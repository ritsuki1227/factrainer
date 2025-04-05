from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import scipy
from numpy import typing as npt

from lightgbm.compat import (  # type: ignore
    dt_DataTable,
    pa_Array,
    pa_ChunkedArray,
    pa_Table,
    pd_DataFrame,
    pd_Series,
)

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
