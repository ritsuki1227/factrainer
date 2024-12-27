from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Self

import lightgbm as lgb
import scipy
from lightgbm.basic import (
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
)
from lightgbm.engine import _LGBM_CustomMetricFunction
from sklearn.model_selection._split import _BaseKFold

from ...domain.base import (
    BaseLearner,
    BaseMlModelConfig,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    DataIndices,
    IndexableDataset,
    Prediction,
    PresettableTrait,
    RawModel,
)
from .equality_checker import LgbDatasetEqualityChecker
from .slicer import (
    LgbDatasetSlicer,
)


class LgbTrainConfig(BaseTrainConfig):
    params: dict[str, Any]
    num_boost_round: int = 100
    valid_names: list[str] | None = None
    feval: _LGBM_CustomMetricFunction | list[_LGBM_CustomMetricFunction] | None = None
    init_model: str | Path | lgb.Booster | None = None
    feature_name: _LGBM_FeatureNameConfiguration = "auto"
    categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto"
    keep_training_booster: bool = False
    callbacks: list[Callable[..., Any]] | None = None


class LgbPredConfig(BasePredictConfig): ...


class LgbDataset(IndexableDataset):
    dataset: lgb.Dataset

    def get_indices(
        self, k_fold: _BaseKFold
    ) -> Generator[tuple[DataIndices, DataIndices], None, None]:
        for train_index, val_index in k_fold.split(self.dataset.data):  # type: ignore
            yield train_index.tolist(), val_index.tolist()

    def split(
        self, train_index: DataIndices, val_index: DataIndices, test_index: DataIndices
    ) -> tuple["LgbDataset", "LgbDataset", "LgbDataset"]:
        train_dataset = LgbDatasetSlicer(reference=None).slice(
            self.dataset, train_index
        )
        val_dataset = LgbDatasetSlicer(reference=train_dataset).slice(
            self.dataset, val_index
        )
        test_dataset = LgbDatasetSlicer(reference=train_dataset).slice(
            self.dataset, val_index
        )
        return (
            LgbDataset(dataset=train_dataset),
            LgbDataset(dataset=val_dataset),
            LgbDataset(dataset=test_dataset),
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LgbDataset):
            return False
        return LgbDatasetEqualityChecker().check(self.dataset, value.dataset)


class LgbModel(RawModel):
    model: lgb.Booster


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


class LgbPredictor(BasePredictor[LgbDataset, LgbModel, LgbPredConfig]):
    def predict(
        self, dataset: LgbDataset, model: LgbModel, config: LgbPredConfig | None
    ) -> Prediction:
        y_pred = model.model.predict(dataset.dataset)
        if isinstance(y_pred, list):
            raise NotImplementedError
        elif isinstance(y_pred, scipy.sparse.spmatrix):
            raise NotImplementedError
        return y_pred


class LgbConfig(
    BaseMlModelConfig[LgbDataset, LgbModel, LgbTrainConfig, LgbPredConfig],
    PresettableTrait[LgbDataset, LgbModel, LgbTrainConfig, LgbPredConfig],
):
    learner: LgbLearner
    predictor: LgbPredictor
    train_config: LgbTrainConfig
    pred_config: LgbPredConfig | None = None

    @classmethod
    def create(
        cls, train_config: LgbTrainConfig, pred_config: LgbPredConfig | None = None
    ) -> Self:
        return cls(
            learner=LgbLearner(),
            predictor=LgbPredictor(),
            train_config=train_config,
            pred_config=pred_config,
        )
