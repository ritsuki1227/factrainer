from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from factrainer.core import CvModelContainer, SingleModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
from numpy import typing as npt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split


@pytest.mark.flaky(reruns=3, reruns_delay=2, only_rerun=["HTTPError"])
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


@pytest.mark.flaky(reruns=3, reruns_delay=2, only_rerun=["HTTPError"])
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
    model = CvModelContainer(config, k_fold, n_jobs_train=4, n_jobs_predict=4)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = r2_score(target, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=2, only_rerun=["HTTPError"])
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
    model = CvModelContainer(config, k_fold, n_jobs_train=4, n_jobs_predict=4)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = accuracy_score(target, y_pred > 0.5)

    assert (metric > 0.8) and (metric < 0.85)


@pytest.mark.flaky(reruns=3, reruns_delay=2, only_rerun=["HTTPError"])
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
