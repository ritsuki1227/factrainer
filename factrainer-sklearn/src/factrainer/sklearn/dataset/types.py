from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from numpy import typing as npt

if TYPE_CHECKING:
    import pandas as pd
    from pandas._typing import Axis


@runtime_checkable
class PdDataFrameProtocol(Protocol):
    def take(
        self, indices: list[int], axis: Axis = 0, **kwargs: Any
    ) -> "PdDataFrameProtocol": ...


@runtime_checkable
class PdSeriesProtocol(Protocol):
    def take(
        self, indices: list[int], axis: Axis = 0, **kwargs: Any
    ) -> "PdSeriesProtocol": ...


if TYPE_CHECKING:
    type PdDataFrameLike = pd.DataFrame
    type PdSeriesLike = pd.Series
else:
    type PdDataFrameLike = PdDataFrameProtocol
    type PdSeriesLike = PdSeriesProtocol

type MatrixLike = npt.NDArray[Any] | PdDataFrameLike
type VectorLike = npt.NDArray[Any] | PdSeriesLike
