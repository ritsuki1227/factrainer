from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from factrainer.core import (
    CvModelContainer,
)
from factrainer.sklearn import (
    SklearnDataset,
    SklearnModelConfig,
    SklearnTrainConfig,
)
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model_regression(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
    dataset = SklearnDataset(X=features, y=target)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=RandomForestRegressor(random_state=100, n_jobs=-1)
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = r2_score(target, y_pred)

    assert_allclose(metric, 0.8, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model_classification(
    iris_data: tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]],
) -> None:
    features, target = iris_data
    dataset = SklearnDataset(X=features, y=target)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(estimator=LogisticRegression()),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=1)
    metric = f1_score(target, y_pred, average="micro")

    assert_allclose(metric, 0.95, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_pandas(
    iris_data: tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]],
) -> None:
    features, target = iris_data
    dataset = SklearnDataset(X=pd.DataFrame(features), y=pd.Series(target))
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(estimator=LogisticRegression())
    )

    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=4)
    y_pred = model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=1)
    metric = f1_score(target, y_pred, average="micro")

    assert_allclose(metric, 0.95, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_polars(
    iris_data: tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]],
) -> None:
    features, target = iris_data
    dataset = SklearnDataset(X=pl.DataFrame(features), y=pl.Series(target))
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(estimator=LogisticRegression())
    )

    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=4)
    y_pred = model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=1)
    metric = f1_score(target, y_pred, average="micro")

    assert_allclose(metric, 0.95, atol=2.5e-02)
