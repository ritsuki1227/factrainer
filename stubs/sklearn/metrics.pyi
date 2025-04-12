from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy import typing as npt

def r2_score(
    y_true: Sequence[Any] | npt.NDArray[np.number[Any]],
    y_pred: Sequence[Any] | npt.NDArray[np.number[Any]],
    *,
    sample_weight: Sequence[Any] | None = None,
    multioutput: str = "uniform_average",
    force_finite: bool = True,
) -> float: ...
