from base.config import (
    BasePredictConfig,
    BaseTrainConfig,
)
from base.dataset import BaseDataset, Prediction
from base.raw_model import RawModel

from ..trait import PredictorTrait, TrainerTrait


class CvMlModel[T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig](
    TrainerTrait[T], PredictorTrait[T, U]
):
    def __init__(self) -> None:
        pass

    def train(self, dataset: T) -> None:
        raise NotImplementedError

    def predict(self, dataset: T) -> Prediction:
        raise NotImplementedError

    @property
    def raw_model(self) -> U:
        raise NotImplementedError
