from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from factrainer.base.dataset import IsImportableInstance
from numpy import typing as npt
from typing_extensions import TypeIs

if TYPE_CHECKING:
    import pandas as pd
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


if TYPE_CHECKING:
    type PdDataFrameLike = pd.DataFrame
    type PdSeriesLike[T] = pd.Series[T]
else:
    type PdDataFrameLike = PdDataFrameProtocol
    type PdSeriesLike[T] = PdSeriesProtocol[T]

type MatrixLike = npt.NDArray[Any] | PdDataFrameLike
type VectorLike = npt.NDArray[Any] | PdSeriesLike[Any]


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
