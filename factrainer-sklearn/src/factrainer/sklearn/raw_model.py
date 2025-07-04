from typing import Any, Protocol, Self, runtime_checkable

from factrainer.base.dataset import Prediction
from factrainer.base.raw_model import RawModel
from numpy import typing as npt

from .dataset.types import MatrixLike, VectorLike


@runtime_checkable
class BaseEstimatorProtocol(Protocol):
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Self: ...


@runtime_checkable
class Predictable(BaseEstimatorProtocol, Protocol):
    def fit(
        self, X: MatrixLike, y: VectorLike | None, *args: Any, **kwargs: Any
    ) -> Self: ...

    def predict(self, X: npt.ArrayLike, *args: Any, **kwargs: Any) -> Prediction: ...


@runtime_checkable
class ProbPredictable(BaseEstimatorProtocol, Protocol):
    def fit(
        self, X: MatrixLike, y: VectorLike | None, *args: Any, **kwargs: Any
    ) -> Self: ...

    def predict_proba(
        self, X: npt.ArrayLike, *args: Any, **kwargs: Any
    ) -> Prediction: ...


class SklearnModel(RawModel):
    """Wrapper for trained scikit-learn model.

    This class wraps a trained scikit-learn estimator instance to provide
    a consistent interface within the factrainer framework.

    Attributes
    ----------
    estimator : Predictable | ProbPredictable
        The trained scikit-learn estimator instance.
    """

    estimator: Predictable | ProbPredictable
