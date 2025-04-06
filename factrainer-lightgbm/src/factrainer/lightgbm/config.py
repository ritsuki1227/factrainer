from pathlib import Path
from typing import Any, Callable, Self

import scipy
from factrainer.base.config import (
    BaseLearner,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    MlModelConfig,
)
from factrainer.base.dataset import Prediction

import lightgbm as lgb
from lightgbm.engine import _LGBM_CustomMetricFunction

from .dataset.dataset import LgbDataset
from .raw_model import LgbModel


class LgbTrainConfig(BaseTrainConfig):
    params: dict[str, Any]
    num_boost_round: int = 100
    valid_names: list[str] | None = None
    feval: _LGBM_CustomMetricFunction | list[_LGBM_CustomMetricFunction] | None = None
    init_model: str | Path | lgb.Booster | None = None
    keep_training_booster: bool = False
    callbacks: list[Callable[..., Any]] | None = None


class LgbPredictConfig(BasePredictConfig): ...


class LgbLearner(BaseLearner[LgbDataset, LgbModel, LgbTrainConfig]):
    def train(
        self,
        train_dataset: LgbDataset,
        val_dataset: LgbDataset | None,
        config: LgbTrainConfig,
    ) -> LgbModel:
        return LgbModel(
            model=lgb.train(
                **dict(config),
                train_set=train_dataset.dataset,
                valid_sets=[val_dataset.dataset] if val_dataset else None,
            )
        )


class LgbPredictor(BasePredictor[LgbDataset, LgbModel, LgbPredictConfig]):
    def predict(
        self, dataset: LgbDataset, model: LgbModel, config: LgbPredictConfig | None
    ) -> Prediction:
        y_pred = model.model.predict(dataset.dataset.data)
        if isinstance(y_pred, list):
            raise NotImplementedError
        elif isinstance(y_pred, scipy.sparse.spmatrix):
            raise NotImplementedError
        return y_pred


class LgbModelConfig(
    MlModelConfig[LgbDataset, LgbModel, LgbTrainConfig, LgbPredictConfig],
):
    learner: LgbLearner
    predictor: LgbPredictor
    train_config: LgbTrainConfig
    pred_config: LgbPredictConfig | None = None

    @classmethod
    def create(
        cls, train_config: LgbTrainConfig, pred_config: LgbPredictConfig | None = None
    ) -> Self:
        return cls(
            learner=LgbLearner(),
            predictor=LgbPredictor(),
            train_config=train_config,
            pred_config=pred_config,
        )
