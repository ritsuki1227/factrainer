from typing import Self

from numpy import typing as npt

class LinearRegression:
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self: ...
