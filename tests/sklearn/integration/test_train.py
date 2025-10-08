from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from factrainer.core import CvModelContainer, SingleModelContainer
from factrainer.sklearn import (
    SklearnDataset,
    SklearnModelConfig,
    SklearnTrainConfig,
)
from factrainer.sklearn.raw_model import Predictable
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


def test_cv_model_regression(
    simulated_regression_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = simulated_regression_data
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
    metric = model.evaluate(target, y_pred, r2_score)

    assert_allclose(metric, 0.67, atol=2.5e-02)
    with pytest.raises(NotFittedError):
        cast(Predictable, config.train_config.estimator).predict(np.array([]))


def test_cv_model_parallel(
    simulated_regression_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = simulated_regression_data
    dataset = SklearnDataset(X=features, y=target)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=RandomForestRegressor(random_state=100, n_jobs=-1)
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=2)
    y_pred = model.predict(dataset)
    metric = model.evaluate(target, y_pred, r2_score)

    assert_allclose(metric, 0.67, atol=2.5e-02)


def test_single_model(
    simulated_regression_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = simulated_regression_data
    train_X, test_X, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=1,
    )
    train_dataset = SklearnDataset(X=train_X, y=train_y)
    test_dataset = SklearnDataset(X=test_X, y=test_y)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=RandomForestRegressor(random_state=100, n_jobs=-1)
        ),
    )
    model = SingleModelContainer(config)
    model.train(train_dataset)
    y_pred = model.predict(test_dataset)
    metric = model.evaluate(test_y, y_pred, r2_score)

    assert_allclose(metric, 0.65, atol=2.5e-02)
