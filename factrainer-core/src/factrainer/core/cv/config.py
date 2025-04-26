from enum import Enum, auto
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from factrainer.base.config import (
    BaseLearner,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
)
from factrainer.base.dataset import BaseDataset, IndexableDataset, Prediction
from factrainer.base.raw_model import RawModel
from joblib import Parallel, delayed

from .dataset import IndexedDatasets
from .raw_model import RawModels


class CvLearner[T: IndexableDataset, U: RawModel, V: BaseTrainConfig](
    BaseLearner[IndexedDatasets[T], RawModels[U], V]
):
    def __init__(self, learner: BaseLearner[T, U, V]) -> None:
        self._learner = learner

    def train(
        self,
        train_dataset: IndexedDatasets[T],
        val_dataset: IndexedDatasets[T] | None,
        config: V,
        n_jobs: int | None = None,
    ) -> RawModels[U]:
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
        return RawModels(models=models)


class PredMode(Enum):
    OOF_PRED = auto()
    AVG_ENSEMBLE = auto()


class OutOfFoldPredictor[T: IndexableDataset, U: RawModel, W: BasePredictConfig](
    BasePredictor[IndexedDatasets[T], RawModels[U], W]
):
    def __init__(self, predictor: BasePredictor[T, U, W]) -> None:
        self._predictor = predictor

    def predict(
        self,
        dataset: IndexedDatasets[T],
        raw_model: RawModels[U],
        config: W | None,
        n_jobs: int | None,
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


class AverageEnsemblePredictor[T: BaseDataset, U: RawModel, W: BasePredictConfig](
    BasePredictor[T, RawModels[U], W]
):
    def __init__(self, predictor: BasePredictor[T, U, W]) -> None:
        self._predictor = predictor

    def predict(
        self,
        dataset: T,
        raw_model: RawModels[U],
        config: W | None,
        n_jobs: int | None,
    ) -> Prediction:
        y_preds = Parallel(n_jobs=n_jobs)(
            delayed(self._predictor.predict)(dataset, _model, config)
            for _model in raw_model.models
        )
        return cast(Prediction, np.array(y_preds).mean(axis=0))
