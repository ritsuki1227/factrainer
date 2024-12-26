from __future__ import annotations

from .base import (
    BaseDataset,
    BaseLearner,
    BaseMlModel,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    NumericNDArray,
    RawModel,
)


class SingleMlModel[
    T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](BaseMlModel[T, U]):
    def __init__(
        self,
        train_config: V,
        learner: BaseLearner[T, U, V],
        predictor: BasePredictor[T, U, W],
        pred_config: W | None = None,
    ) -> None:
        self.train_config = train_config
        self._learner = learner
        self._predictor = predictor
        self.pred_config = pred_config

    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        self._model = self._learner.train(train_dataset, val_dataset, self.train_config)

    def predict(self, dataset: T) -> NumericNDArray:
        return self._predictor.predict(dataset, self.model, self.pred_config)

    @property
    def model(self) -> U:
        return self._model
