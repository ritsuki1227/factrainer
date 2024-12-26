from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
from lightgbm.basic import (
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
)
from lightgbm.engine import _LGBM_CustomMetricFunction
from sklearn.model_selection._split import _BaseKFold

from ...domain.base import (
    BaseLearner,
    BasePredictor,
    BaseTrainConfig,
    DataIndices,
    IndexableDataset,
    NumericNDArray,
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
        dataset: LgbDataset,
        config: LgbTrainConfig,
    ) -> LgbModel:
        raise NotImplementedError
        # return LgbModel(
        #     model=lgb.train(
        #         **dict(config),
        #         train_set=train_dataset.dataset,
        #         valid_sets=[val_dataset.dataset] if val_dataset else None,
        #     )
        # )


class LgbPredictor(BasePredictor[LgbDataset, LgbModel]):
    def predict(self, dataset: LgbDataset, model: LgbModel) -> NumericNDArray:
        raise NotImplementedError
        # return model.model.predict(dataset.dataset)


# # class LgbSingleTrainerFactory(BaseTrainerFactory[LgbDataset, LgbTrainConfig]):
# #     @classmethod
# #     def create(cls, config: LgbTrainConfig) -> BaseTrainer[LgbDataset]:
# #         return SingleTrainer(config, LgbLearner(), LgbPredictor())
