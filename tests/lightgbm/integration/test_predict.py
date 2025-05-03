from __future__ import annotations

import tempfile
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pytest
from factrainer.core import (
    CvModelContainer,
    PredMode,
)
from factrainer.lightgbm import (
    LgbDataset,
    LgbModel,
    LgbModelConfig,
    LgbPredictConfig,
    LgbTrainConfig,
)
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_pred_config(
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
        pred_config=LgbPredictConfig(num_iteration=2),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = r2_score(target, y_pred)

    assert_allclose(metric, 0.22, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_set_pred_config_after_trainig(
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
        pred_config=LgbPredictConfig(num_iteration=2),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    metric_underfit = r2_score(target, model.predict(dataset))
    model.pred_config = LgbPredictConfig(num_iteration=None)
    metric = r2_score(target, model.predict(dataset))

    assert_allclose(metric_underfit, 0.22, atol=2.5e-02)
    assert_allclose(metric, 0.84, atol=2.5e-02)


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
            params={"objective": "regression", "seed": 1, "deterministic": True},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(train_dataset, n_jobs=4)
    y_pred = model.predict(test_dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
    metric = r2_score(test_y, y_pred)

    assert_allclose(metric, 0.84, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model_picklable(
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
    model.train(dataset, n_jobs=4)
    with tempfile.NamedTemporaryFile() as fp:
        joblib.dump(model, fp.name)
        loaded_model: CvModelContainer[
            LgbDataset, LgbModel, LgbTrainConfig, LgbPredictConfig
        ] = joblib.load(fp.name)
    y_pred = loaded_model.predict(dataset, n_jobs=4)
    metric = r2_score(target, y_pred)

    assert_allclose(metric, 0.84, atol=2.5e-02)
