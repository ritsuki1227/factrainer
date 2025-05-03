from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from factrainer.base.dataset import IsImportableInstance
from numpy import typing as npt
from typing_extensions import TypeIs

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


@runtime_checkable
class PdDataFrameProtocol(Protocol): ...


@runtime_checkable
class PdSeriesProtocol[T](Protocol): ...


@runtime_checkable
class PlDataFrameProtocol(Protocol): ...


@runtime_checkable
class PlSeriesProtocol(Protocol): ...


if TYPE_CHECKING:
    type PdDataFrameLike = pd.DataFrame
    type PdSeriesLike[T] = pd.Series[T]
    type PlDataFrameLike = pl.DataFrame
    type PlSeriesLike = pl.Series
else:
    type PdDataFrameLike = PdDataFrameProtocol
    type PdSeriesLike[T] = PdSeriesProtocol[T]
    type PlDataFrameLike = PlDataFrameProtocol
    type PlSeriesLike = PlSeriesProtocol

type MatrixLike = npt.NDArray[Any] | PdDataFrameLike | PlDataFrameLike
type VectorLike = npt.NDArray[Any] | PdSeriesLike[Any] | PlSeriesLike


class IsPdDataFrame(IsImportableInstance[PdDataFrameLike]):
    def is_instance(self, obj: Any) -> TypeIs[PdDataFrameLike]:
        try:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)
        except ImportError:
            return False


class IsPdSeries(IsImportableInstance[PdSeriesLike[Any]]):
    def is_instance(self, obj: Any) -> TypeIs[PdSeriesLike[Any]]:
        try:
            import pandas as pd

            return isinstance(obj, pd.Series)
        except ImportError:
            return False


class IsPlDataFrame(IsImportableInstance[PlDataFrameLike]):
    def is_instance(self, obj: Any) -> TypeIs[PlDataFrameLike]:
        try:
            import polars as pl

            return isinstance(obj, pl.DataFrame)
        except ImportError:
            return False


class IsPlSeries(IsImportableInstance[PlSeriesLike]):
    def is_instance(self, obj: Any) -> TypeIs[PlSeriesLike]:
        try:
            import polars as pl

            return isinstance(obj, pl.Series)
        except ImportError:
            return False
