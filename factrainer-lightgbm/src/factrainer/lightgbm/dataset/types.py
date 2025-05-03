from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from factrainer.base.dataset import IsImportableInstance
from typing_extensions import TypeIs

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


@runtime_checkable
class PdDataFrameProtocol(Protocol): ...


@runtime_checkable
class PdSeriesProtocol[T](Protocol): ...


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
