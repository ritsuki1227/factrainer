from typing import Any, Self

import numpy as np
import numpy.typing as npt
from factrainer.base.config import (
    BaseLearner,
    BaseMlModelConfig,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
)
from factrainer.base.dataset import IndexableDataset, Prediction
from factrainer.base.raw_model import RawModel
from joblib import Parallel, delayed

from .dataset import IndexedDatasets
from .raw_model import CvRawModels


class CvLearner[T: IndexableDataset, U: RawModel, V: BaseTrainConfig](
    BaseLearner[IndexedDatasets[T], CvRawModels[U], V]
):
    def __init__(self, learner: BaseLearner[T, U, V]) -> None:
        self._learner = learner

    def train(
        self,
        train_dataset: IndexedDatasets[T],
        val_dataset: IndexedDatasets[T] | None,
        config: V,
        n_jobs: int | None = None,
    ) -> CvRawModels[U]:
        if val_dataset is not None:
            models = Parallel(n_jobs=n_jobs)(
                delayed(self._learner.train)(train.data, val.data, config)
                for train, val in zip(train_dataset.datasets, val_dataset.datasets)
            )
        else:
            models = Parallel(n_jobs=n_jobs)(
                delayed(self._learner.train)(train.data, None, config)
                for train in train_dataset.datasets
            )
        return CvRawModels(models=models)


class CvPredictor[T: IndexableDataset, U: RawModel, W: BasePredictConfig](
    BasePredictor[IndexedDatasets[T], CvRawModels[U], W]
):
    def __init__(self, predictor: BasePredictor[T, U, W]) -> None:
        self._predictor = predictor

    def predict(
        self,
        dataset: IndexedDatasets[T],
        raw_model: CvRawModels[U],
        config: W | None,
        n_jobs: int | None = None,
    ) -> Prediction:
        y_preds = Parallel(n_jobs=n_jobs)(
            delayed(self._predictor.predict)(_dataset.data, _model, config)
            for _model, _dataset in zip(raw_model.models, dataset.datasets)
        )
        y_oof_pred = self._init_pred(len(dataset), y_preds[0])
        for y_pred, _dataset in zip(y_preds, dataset.datasets):
            y_oof_pred[_dataset.index] = y_pred
        return y_oof_pred

    def _init_pred(self, total_length: int, y_pred: npt.NDArray[Any]) -> Prediction:
        return np.empty(
            tuple([total_length] + list(y_pred.shape[1:])), dtype=y_pred.dtype
        )


class CvMlModelConfig[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseMlModelConfig[IndexedDatasets[T], CvRawModels[U], V, W]):
    learner: CvLearner[T, U, V]
    predictor: CvPredictor[T, U, W]

    @classmethod
    def from_config(cls, config: BaseMlModelConfig[T, U, V, W]) -> Self:
        return cls(
            learner=CvLearner(config.learner),
            predictor=CvPredictor(config.predictor),
            train_config=config.train_config,
            pred_config=config.pred_config,
        )
