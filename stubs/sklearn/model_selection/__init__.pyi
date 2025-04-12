from typing import Any

import numpy as np
import numpy.typing as npt

from ._split import _BaseKFold

class _UnsupportedGroupCVMixin: ...

class KFold(_UnsupportedGroupCVMixin, _BaseKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...

def train_test_split(
    X: npt.NDArray[np.number[Any]],
    y: npt.NDArray[np.number[Any]] | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[
    npt.NDArray[np.number[Any]],
    npt.NDArray[np.number[Any]],
    npt.NDArray[np.number[Any]],
    npt.NDArray[np.number[Any]],
]: ...
