from __future__ import annotations

from .base import (
    BaseDataset,
    BaseMlModel,
    BaseMlModelConfig,
    BasePredictConfig,
    BaseTrainConfig,
    NumericNDArray,
    RawModel,
)


class SingleMlModel[
    T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](BaseMlModel[T, U]):
    def __init__(
        self,
        config: BaseMlModelConfig[T, U, V, W],
    ) -> None:
        self.config = config

    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        self._model = self.config.learner.train(
            train_dataset, val_dataset, self.config.train_config
        )

    def predict(self, dataset: T) -> NumericNDArray:
        return self.config.predictor.predict(
            dataset, self.model, self.config.pred_config
        )

    @property
    def model(self) -> U:
        return self._model


# class SingleMlModel[
#     T: BaseDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
# ](BaseMlModel[T, U]):
#     def __init__(
#         self,
#         train_config: V,
#         learner: BaseLearner[T, U, V],
#         predictor: BasePredictor[T, U, W],
#         pred_config: W | None = None,
#     ) -> None:
#         self.train_config = train_config
#         self._learner = learner
#         self._predictor = predictor
#         self.pred_config = pred_config

#     def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
#         self._model = self._learner.train(train_dataset, val_dataset, self.train_config)

#     def predict(self, dataset: T) -> NumericNDArray:
#         return self._predictor.predict(dataset, self.model, self.pred_config)

#     @property
#     def model(self) -> U:
#         return self._model
