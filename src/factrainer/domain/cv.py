from __future__ import annotations

import os
from typing import Self, Sequence

import numpy as np
from sklearn.model_selection._split import _BaseKFold

from .base import (
    BaseDataset,
    BaseLearner,
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
    BaseLearner[SplittedDatasets[T], CvRawModels[U], V]
):
    def __init__(
        self, learner: BaseLearner[T, U, V], n_jobs: int | None = None
    ) -> None:
        self._learner = learner
        self._n_jobs = n_jobs

    def train(self, dataset: SplittedDatasets[T], config: V) -> CvRawModels[U]:
        models = []
        for _dataset in dataset.datasets:
            model = self._learner.train(_dataset.train.data, config)
            models.append(model)
        return CvRawModels(models=models)

    @property
    def n_jobs(self) -> int | None:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int | None) -> None:
        self._n_jobs = n_jobs


class CvPredictor[T: IndexableDataset, U: RawModel](
    BasePredictor[SplittedDatasets[T], CvRawModels[U]]
):
    def __init__(
        self, predictor: BasePredictor[T, U], n_jobs: int | None = None
    ) -> None:
        self._predictor = predictor
        self._n_jobs = n_jobs

    def predict(
        self, dataset: SplittedDatasets[T], model: CvRawModels[U]
    ) -> NumericNDArray:
        for i, (_model, _dataset) in enumerate(zip(model.models, dataset.datasets)):
            y_pred = self._predictor.predict(_dataset.test.data, _model)
            if i == 0:
                y_oof_pred = self._init_pred(len(dataset), y_pred.shape)
            y_oof_pred[_dataset.test.indices] = y_pred
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


class CvMlModel[T: IndexableDataset, U: RawModel, V: BaseTrainConfig](
    SingleMlModel[SplittedDatasets[T], CvRawModels[U], V]
):
    def __init__(
        self,
        config: V,
        learner: BaseLearner[T, U, V],
        predictor: BasePredictor[T, U],
        n_jobs_train: int | None = None,
        n_jobs_predict: int | None = None,
    ) -> None:
        super().__init__(
            config,
            CvLearner(learner, n_jobs_train),
            CvPredictor(predictor, n_jobs_predict),
        )

    def train(self, dataset: SplittedDatasets[T]) -> None:
        self.datasets = dataset
        super().train(self.datasets)

    def predict(self, dataset: SplittedDatasets[T] | None = None) -> NumericNDArray:
        if dataset is None:
            super().predict(self.datasets)
        raise NotImplementedError

    @property
    def datasets(self) -> SplittedDatasets[T]:
        return self._datasets

    @datasets.setter
    def datasets(self, datasets: SplittedDatasets[T]) -> None:
        self._datasets = datasets

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


# class CvMlModel[T: Dataset, U: RawModel, V: BaseTrainConfig](
#     BaseMlModel[T, CvRawModels[U]]
# ):
#     def __init__(
#         self,
#         k_fold: BaseCrossValidator,
#         config: V,
#         learner: BaseLearner[T, U, V],
#         predictor: BasePredictor[T, U],
#         n_jobs_train: int | None = None,
#         n_jobs_predict: int | None = None,
#     ) -> None:
#         self._k_fold = k_fold
#         self.config = config
#         self._learner = CvLearner(learner, n_jobs_train)
#         self._predictor = CvPredictor(predictor, n_jobs_predict)

#     def train(self, dataset: T) -> None:
#         self.datasets = IndexedTrainValDatasets.create(self._k_fold, dataset)
#         self._model = self._learner.train(self.datasets, self.config)

#     def predict(self, dataset: T | None = None) -> NumericNDArray:
#         if dataset is None:
#             return self._predictor.predict(self.datasets, self.model)
#         raise NotImplementedError

#     @property
#     def model(self) -> CvRawModels[U]:
#         return self._model

#     @property
#     def datasets(self) -> IndexedTrainValDatasets[T]:
#         return self._datasets

#     @datasets.setter
#     def datasets(self, datasets: IndexedTrainValDatasets[T]) -> None:
#         self._datasets = datasets

#     @property
#     def n_jobs_train(self) -> int | None:
#         return self._learner.n_jobs

#     @n_jobs_train.setter
#     def n_jobs_train(self, n_jobs: int | None) -> None:
#         self._learner.n_jobs = n_jobs

#     @property
#     def n_jobs_predict(self) -> int | None:
#         return self._predictor.n_jobs

#     @n_jobs_predict.setter
#     def n_jobs_predict(self, n_jobs: int | None) -> None:
#         self._predictor.n_jobs = n_jobs
