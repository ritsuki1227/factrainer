import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.utils._bunch import Bunch


@pytest.fixture(scope="package")
def _california_housing_data() -> Bunch:
    return fetch_california_housing()
