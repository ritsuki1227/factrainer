from __future__ import annotations

from collections.abc import Generator
from typing import Any

import lightgbm as lgb
from sklearn.model_selection._split import _BaseKFold

from ...domain.base import BaseTrainConfig, DataIndices, IndexableDataset
from .equality_checker import LgbDatasetEqualityChecker
from .slicer import (
    LgbDatasetSlicer,
)


class LgbTrainConfig(BaseTrainConfig):
    params: dict[str, Any]


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


# class LgbModel(RawModel):
#     model: lgb.Booster


# class LgbLearner(BaseLearner[LgbDataset, LgbModel, LgbTrainConfig]):
#     def train(self, dataset: LgbDataset, config: LgbTrainConfig) -> LgbModel:
#         raise NotImplementedError
#         # return LgbModel(model=lgb.train(config.params, dataset.dataset))


# class LgbPredictor(BasePredictor[LgbDataset, LgbModel]):
#     def predict(self, dataset: LgbDataset, model: LgbModel) -> NumericNDArray:
#         raise NotImplementedError
#         # return model.model.predict(dataset.dataset)


# # class LgbSingleTrainerFactory(BaseTrainerFactory[LgbDataset, LgbTrainConfig]):
# #     @classmethod
# #     def create(cls, config: LgbTrainConfig) -> BaseTrainer[LgbDataset]:
# #         return SingleTrainer(config, LgbLearner(), LgbPredictor())
