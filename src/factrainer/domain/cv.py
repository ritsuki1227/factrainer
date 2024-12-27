from __future__ import annotations

import os
from typing import Self, Sequence

import numpy as np
from sklearn.model_selection._split import _BaseKFold

from .base import (
    BaseDataset,
    BaseLearner,
    BaseMlModelConfig,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    DataIndices,
    IndexableDataset,
    NumericNDArray,
    RawModel,
)
from .single import SingleMlModel


class IndexedDataset[T: IndexableDataset](BaseDataset):
    indices: DataIndices
    data: T

    def __len__(self) -> int:
        return len(self.indices)


class IndexedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[IndexedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])


class SplittedDataset[T: IndexableDataset](BaseDataset):
    train: IndexedDataset[T]
    val: IndexedDataset[T]
    test: IndexedDataset[T]


class SplittedDatasets[T: IndexableDataset](BaseDataset):
    datasets: Sequence[SplittedDataset[T]]

    def __len__(self) -> int:
        return sum([len(dataset.test) for dataset in self.datasets])

    @property
    def train(self) -> IndexedDatasets[T]:
        return IndexedDatasets(datasets=[dataset.train for dataset in self.datasets])

    @property
    def val(self) -> IndexedDatasets[T]:
        return IndexedDatasets(datasets=[dataset.val for dataset in self.datasets])

    @property
    def test(self) -> IndexedDatasets[T]:
        return IndexedDatasets(datasets=[dataset.test for dataset in self.datasets])

    @classmethod
    def create(
        cls, k_fold: _BaseKFold, dataset: T, share_holdouts: bool = True
    ) -> Self:
        datasets = []
        for train_index, val_index in dataset.get_indices(k_fold):
            if share_holdouts:
                test_index = val_index
            else:
                raise NotImplementedError
            train_dataset, val_dataset, test_dataset = dataset.split(
                train_index, val_index, test_index
            )
            datasets.append(
                SplittedDataset(
                    train=IndexedDataset(indices=train_index, data=train_dataset),
                    val=IndexedDataset(indices=val_index, data=val_dataset),
                    test=IndexedDataset(indices=test_index, data=test_dataset),
                )
            )
        return cls(datasets=datasets)


class CvRawModels[U: RawModel](RawModel):
    models: Sequence[U]


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs:
        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        return n_jobs
    cpus = os.cpu_count()
    if cpus is None:
        raise ValueError("Failed to determine the number of CPUs")
    return cpus


class CvLearner[T: IndexableDataset, U: RawModel, V: BaseTrainConfig](
    BaseLearner[IndexedDatasets[T], CvRawModels[U], V]
):
    def __init__(
        self, learner: BaseLearner[T, U, V], n_jobs: int | None = None
    ) -> None:
        self._learner = learner
        self._n_jobs = n_jobs

    def train(
        self,
        train_dataset: IndexedDatasets[T],
        val_dataset: IndexedDatasets[T] | None,
        config: V,
    ) -> CvRawModels[U]:
        models = []
        if val_dataset is not None:
            for train, val in zip(train_dataset.datasets, val_dataset.datasets):
                model = self._learner.train(train.data, val.data, config)
                models.append(model)
        else:
            for train in train_dataset.datasets:
                model = self._learner.train(train.data, None, config)
                models.append(model)
        return CvRawModels(models=models)

    @property
    def n_jobs(self) -> int | None:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int | None) -> None:
        self._n_jobs = n_jobs


class CvPredictor[T: IndexableDataset, U: RawModel, W: BasePredictConfig](
    BasePredictor[IndexedDatasets[T], CvRawModels[U], W]
):
    def __init__(
        self, predictor: BasePredictor[T, U, W], n_jobs: int | None = None
    ) -> None:
        self._predictor = predictor
        self._n_jobs = n_jobs

    def predict(
        self, dataset: IndexedDatasets[T], model: CvRawModels[U], config: W | None
    ) -> NumericNDArray:
        for i, (_model, _dataset) in enumerate(zip(model.models, dataset.datasets)):
            y_pred = self._predictor.predict(_dataset.data, _model, config)
            if i == 0:
                y_oof_pred = self._init_pred(len(dataset), y_pred.shape)
            y_oof_pred[_dataset.indices] = y_pred
        return y_oof_pred

    def _init_pred(
        self, total_length: int, y_pred_shape: tuple[int, ...]
    ) -> NumericNDArray:
        return np.empty(tuple([total_length] + list(y_pred_shape[1:])))

    @property
    def n_jobs(self) -> int | None:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int | None) -> None:
        self._n_jobs = n_jobs


class CvMlModelConfig[
    T: IndexableDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](BaseMlModelConfig[IndexedDatasets[T], CvRawModels[U], V, W]):
    @classmethod
    def from_config(
        cls,
        config: BaseMlModelConfig[T, U, V, W],
        n_jobs_train: int | None = None,
        n_jobs_predict: int | None = None,
    ) -> Self:
        return cls(
            train_config=config.train_config,
            learner=CvLearner(config.learner, n_jobs_train),
            predictor=CvPredictor(config.predictor, n_jobs_predict),
            pred_config=config.pred_config,
        )


class CvMlModel[
    T: IndexableDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
](SingleMlModel[IndexedDatasets[T], CvRawModels[U], V, W]):
    def __init__(
        self,
        config: BaseMlModelConfig[T, U, V, W],
        n_jobs_train: int | None = None,
        n_jobs_predict: int | None = None,
    ) -> None:
        super().__init__(
            CvMlModelConfig.from_config(config, n_jobs_train, n_jobs_predict)
        )

    def train(
        self,
        train_dataset: IndexedDatasets[T],
        val_dataset: IndexedDatasets[T] | None = None,
    ) -> None:
        self.train_datasets = train_dataset
        self.val_datasets = val_dataset
        super().train(self.train_datasets, self.val_datasets)

    def predict(self, dataset: IndexedDatasets[T]) -> NumericNDArray:
        return super().predict(dataset)

    @property
    def train_datasets(self) -> IndexedDatasets[T]:
        return self._train_datasets

    @train_datasets.setter
    def train_datasets(self, datasets: IndexedDatasets[T]) -> None:
        self._train_datasets = datasets

    @property
    def val_datasets(self) -> IndexedDatasets[T] | None:
        return self._val_datasets

    @val_datasets.setter
    def val_datasets(self, datasets: IndexedDatasets[T] | None) -> None:
        self._val_datasets = datasets

    @property
    def n_jobs_train(self) -> int | None:
        return self._learner.n_jobs  # type: ignore

    @n_jobs_train.setter
    def n_jobs_train(self, n_jobs: int | None) -> None:
        self._learner.n_jobs = n_jobs  # type: ignore

    @property
    def n_jobs_predict(self) -> int | None:
        return self._predictor.n_jobs  # type: ignore

    @n_jobs_predict.setter
    def n_jobs_predict(self, n_jobs: int | None) -> None:
        self._predictor.n_jobs = n_jobs  # type: ignore


# class CvMlModel[
#     T: IndexableDataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig
# ](SingleMlModel[IndexedDatasets[T], CvRawModels[U], V, W]):
#     def __init__(
#         self,
#         train_config: V,
#         learner: BaseLearner[T, U, V],
#         predictor: BasePredictor[T, U, W],
#         pred_config: W | None = None,
#         n_jobs_train: int | None = None,
#         n_jobs_predict: int | None = None,
#     ) -> None:
#         super().__init__(
#             train_config,
#             CvLearner(learner, n_jobs_train),
#             CvPredictor(predictor, n_jobs_predict),
#             pred_config,
#         )

#     def train(
#         self,
#         train_dataset: IndexedDatasets[T],
#         val_dataset: IndexedDatasets[T] | None = None,
#     ) -> None:
#         self.train_datasets = train_dataset
#         self.val_datasets = val_dataset
#         super().train(self.train_datasets, self.val_datasets)

#     def predict(self, dataset: IndexedDatasets[T]) -> NumericNDArray:
#         return super().predict(dataset)

#     @property
#     def train_datasets(self) -> IndexedDatasets[T]:
#         return self._train_datasets

#     @train_datasets.setter
#     def train_datasets(self, datasets: IndexedDatasets[T]) -> None:
#         self._train_datasets = datasets

#     @property
#     def val_datasets(self) -> IndexedDatasets[T] | None:
#         return self._val_datasets

#     @val_datasets.setter
#     def val_datasets(self, datasets: IndexedDatasets[T] | None) -> None:
#         self._val_datasets = datasets

#     @property
#     def n_jobs_train(self) -> int | None:
#         return self._learner.n_jobs  # type: ignore

#     @n_jobs_train.setter
#     def n_jobs_train(self, n_jobs: int | None) -> None:
#         self._learner.n_jobs = n_jobs  # type: ignore

#     @property
#     def n_jobs_predict(self) -> int | None:
#         return self._predictor.n_jobs  # type: ignore

#     @n_jobs_predict.setter
#     def n_jobs_predict(self, n_jobs: int | None) -> None:
#         self._predictor.n_jobs = n_jobs  # type: ignore
