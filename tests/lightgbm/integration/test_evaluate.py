from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pytest
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


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_evaluate_pooling_mode(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
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

    assert_allclose(metric, 0.84, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_evaluate_fold_wise_mode(
    california_housing_data: tuple[
        npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
    ],
) -> None:
    features, target = california_housing_data
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
        assert_allclose(metric, 0.84, atol=2.5e-02)
