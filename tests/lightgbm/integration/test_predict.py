from __future__ import annotations

import tempfile
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
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


def test_cv_pred_config(
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
        pred_config=LgbPredictConfig(num_iteration=2),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred = model.predict(dataset)
    metric = model.evaluate(target, y_pred, r2_score)

    assert_allclose(metric, 0.19, atol=2.5e-02)


def test_cv_set_pred_config_after_training(
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
        pred_config=LgbPredictConfig(num_iteration=2),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    metric_underfit = model.evaluate(target, model.predict(dataset), r2_score)
    model.pred_config = LgbPredictConfig(num_iteration=None)
    metric = model.evaluate(target, model.predict(dataset), r2_score)

    assert_allclose(metric_underfit, 0.19, atol=2.5e-02)
    assert_allclose(metric, 0.87, atol=2.5e-02)


def test_cv_average_ensembling(
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
    metric = model.evaluate(test_y, y_pred, r2_score)

    assert_allclose(metric, 0.87, atol=2.5e-02)


def test_cv_model_picklable(
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
    model.train(dataset, n_jobs=4)
    with tempfile.NamedTemporaryFile() as fp:
        joblib.dump(model, fp.name)
        loaded_model: CvModelContainer[
            LgbDataset, LgbModel, LgbTrainConfig, LgbPredictConfig
        ] = joblib.load(fp.name)
    y_pred = loaded_model.predict(dataset, n_jobs=4)
    metric = loaded_model.evaluate(target, y_pred, r2_score)

    assert_allclose(metric, 0.87, atol=2.5e-02)
