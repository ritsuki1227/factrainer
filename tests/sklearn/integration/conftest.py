from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy import typing as npt
from sklearn.datasets import fetch_california_housing, load_iris


@pytest.fixture(scope="package")
def california_housing_data() -> tuple[
    npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
]:
    data = fetch_california_housing()
    return data["data"], data["target"]  # type: ignore


@pytest.fixture(scope="package")
def iris_data() -> tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]]:
    return load_iris(return_X_y=True)
