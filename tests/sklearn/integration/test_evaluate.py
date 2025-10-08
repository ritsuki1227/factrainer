from __future__ import annotations

from typing import Any

import numpy as np
from factrainer.core import CvModelContainer, EvalMode
from factrainer.sklearn import (
    SklearnDataset,
    SklearnModelConfig,
    SklearnTrainConfig,
)
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def test_evaluate_pooling_mode(
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

    metric = model.evaluate(target, y_pred, r2_score, eval_mode=EvalMode.POOLING)

    assert_allclose(metric, 0.67, atol=2.5e-02)


def test_evaluate_fold_wise_mode(
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

    metrics = model.evaluate(target, y_pred, r2_score, eval_mode=EvalMode.FOLD_WISE)

    for metric in metrics:
        assert_allclose(metric, 0.67, atol=5.0e-02)
