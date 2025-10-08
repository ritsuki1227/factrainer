from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


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
    metric = model.evaluate(
        target, y_pred, lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")
    )

    assert_allclose(metric, 0.95, atol=2.5e-02)


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
    metric = model.evaluate(
        target, y_pred, lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")
    )

    assert_allclose(metric, 0.95, atol=2.5e-02)
