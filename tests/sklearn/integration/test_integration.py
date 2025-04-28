from __future__ import annotations

from typing import Any

import numpy as np
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model(
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

    assert (metric > 0.8) and (metric < 0.85)
