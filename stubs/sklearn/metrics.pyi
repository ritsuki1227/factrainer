from collections.abc import Sequence
from typing import Any

def r2_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    sample_weight: Sequence[Any] | None = None,
    multioutput: str = "uniform_average",
    force_finite: bool = True,
) -> float: ...
