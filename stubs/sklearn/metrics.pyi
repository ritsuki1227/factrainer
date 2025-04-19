from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt

def r2_score(
    y_true: Sequence[Any] | npt.NDArray[np.number[Any]] | pd.DataFrame | pd.Series[Any],
    y_pred: npt.ArrayLike,
    *,
    sample_weight: Sequence[Any] | None = None,
    multioutput: str = "uniform_average",
    force_finite: bool = True,
) -> float: ...
def accuracy_score(
    y_true: Sequence[Any] | npt.NDArray[np.number[Any]] | pd.DataFrame | pd.Series[Any],
    y_pred: npt.ArrayLike,
    *,
    normalize: bool = True,
    sample_weight: Sequence[Any] | None = None,
) -> float: ...
