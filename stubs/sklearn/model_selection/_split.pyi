from abc import ABCMeta
from collections.abc import Generator

import numpy as np
from numpy import typing as npt

class _MetadataRequester: ...
class BaseCrossValidator(_MetadataRequester, metaclass=ABCMeta): ...

class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> Generator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]: ...
