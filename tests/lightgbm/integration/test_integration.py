from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from factrainer.core import (
    CvModelContainer,
    PredMode,
    SingleModelContainer,
    SplittedDatasetsIndices,
)
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
from numpy import typing as npt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = r2_score(target, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model_parallel(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=4)
    y_pred = model.predict(dataset, n_jobs=4)
    metric = r2_score(target, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_pandas(titanic_data: tuple[pd.DataFrame, pd.Series[int]]) -> None:
    features, target = titanic_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "binary"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=4)
    y_pred = model.predict(dataset, n_jobs=4)
    metric = accuracy_score(target, y_pred > 0.5)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_train_val_test_split(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    train_indices, val_indices, test_indices = [], [], []
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    for i, (_train_index, test_index) in enumerate(k_fold.split(features)):
        splitted = train_test_split(_train_index, test_size=0.25, random_state=i + 1)
        train_index, val_index = splitted[0], splitted[1]
        train_indices.append(train_index)
        val_indices.append(val_index)
        test_indices.append(test_index)
    indices = SplittedDatasetsIndices(
        train=train_indices, val=val_indices, test=test_indices
    )
    model = CvModelContainer(config, indices)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = r2_score(target, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_average_ensembling(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    train_X, test_X, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=1,
    )
    train_dataset = LgbDataset(dataset=lgb.Dataset(train_X, train_y))
    test_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(train_dataset, n_jobs=4)
    y_pred = model.predict(test_dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
    metric = r2_score(test_y, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_single_model(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    train_X, test_X, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=1,
    )
    train_dataset = LgbDataset(dataset=lgb.Dataset(train_X, train_y))
    val_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    test_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression"},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    model = SingleModelContainer(config)
    model.train(train_dataset, val_dataset)
    y_pred = model.predict(test_dataset)
    metric = r2_score(test_y, y_pred)

    assert (metric > 0.8) and (metric < 0.85)
