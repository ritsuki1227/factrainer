from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Sequence

from factrainer.base.config import BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import BaseDataset, Prediction, Target
from factrainer.base.raw_model import RawModel

type EvalFunc[T] = Callable[[Target, Prediction], T]


class BaseModelContainer[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
    # X,
](ABC):
    @abstractmethod
    def train(self, train_dataset: T, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, pred_dataset: T, *args: Any, **kwargs: Any) -> Prediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_model(self) -> U:
        raise NotImplementedError

    @abstractmethod
    def evaluate[X](
        self,
        y_true: Target,
        y_pred: Prediction,
        eval_func: EvalFunc[X],
        *args: Any,
        **kwargs: Any,
    ) -> X | Sequence[X]:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_config(self) -> V:
        raise NotImplementedError

    @train_config.setter
    @abstractmethod
    def train_config(self, config: V) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def pred_config(self) -> W:
        raise NotImplementedError

    @pred_config.setter
    @abstractmethod
    def pred_config(self, config: W) -> None:
        raise NotImplementedError
