from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from sklearn.datasets import fetch_california_housing, fetch_openml


@pytest.fixture(scope="package")
def california_housing_data() -> tuple[
    npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
]:
    data = fetch_california_housing()
    return data["data"], data["target"]  # type: ignore


@pytest.fixture(scope="package")
def titanic_data() -> tuple[pd.DataFrame, pd.Series[int]]:
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)
    features = titanic["data"].drop(  # type: ignore
        columns=["name", "ticket", "cabin", "boat", "body", "home.dest"]
    )
    target = titanic["target"].astype("int")  # type: ignore
    return features, target
