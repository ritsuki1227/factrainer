from __future__ import annotations

from .base import (
    BaseDataset,
    BaseLearner,
    BaseMlModel,
    BasePredictor,
    BaseTrainConfig,
    NumericNDArray,
    RawModel,
)


class SingleMlModel[T: BaseDataset, U: RawModel, V: BaseTrainConfig](BaseMlModel[T, U]):
    def __init__(
        self, config: V, learner: BaseLearner[T, U, V], predictor: BasePredictor[T, U]
    ) -> None:
        self.config = config
        self._learner = learner
        self._predictor = predictor

    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        self._model = self._learner.train(train_dataset, val_dataset, self.config)

    def predict(self, dataset: T) -> NumericNDArray:
        return self._predictor.predict(dataset, self.model)

    @property
    def model(self) -> U:
        return self._model
