from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy import typing as npt
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_regression,
)


@pytest.fixture(scope="package")
def iris_data() -> tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]]:
    return load_iris(return_X_y=True)


@pytest.fixture(scope="package")
def breast_cancer_data() -> tuple[
    npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
]:
    return load_breast_cancer(return_X_y=True)


@pytest.fixture(scope="package")
def simulated_regression_data() -> tuple[
    npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
]:
    return make_regression(n_samples=1000, random_state=1)
