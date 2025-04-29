from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

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
def f1_score(
    y_true: Sequence[Any] | npt.NDArray[np.number[Any]] | pd.DataFrame | pd.Series[Any],
    y_pred: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
    pos_label: str | int = 1,
    average: Literal["micro", "macro", "samples", "weighted", "binary"]
    | None = "binary",
    sample_weight: npt.ArrayLike | None = None,
    zero_division: int | Literal["warn"] = "warn",
) -> float: ...

#   *,
#     labels: ArrayLike | None = None,
#     pos_label: str | int = 1,
#     average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] | None = "binary",
#     sample_weight: ArrayLike | None = None,
#     zero_division: int | Literal['warn'] = "warn"
# ) -> (Float | ndarray)
