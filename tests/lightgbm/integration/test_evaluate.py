from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
from factrainer.core import CvModelContainer, EvalMode
from factrainer.lightgbm import (
    LgbDataset,
    LgbModelConfig,
    LgbTrainConfig,
)
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def test_evaluate_pooling_mode(
    simulated_regression_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = simulated_regression_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression", "seed": 1, "deterministic": True},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)

    metric = model.evaluate(target, y_pred, r2_score, eval_mode=EvalMode.POOLING)

    assert_allclose(metric, 0.87, atol=2.5e-02)


def test_evaluate_fold_wise_mode(
    simulated_regression_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = simulated_regression_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "regression", "seed": 1, "deterministic": True},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)

    metrics = model.evaluate(target, y_pred, r2_score, eval_mode=EvalMode.FOLD_WISE)

    for metric in metrics:
        assert_allclose(metric, 0.87, atol=2.5e-02)
