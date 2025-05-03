from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import pytest
from factrainer.core import (
    CvModelContainer,
)
from factrainer.lightgbm import (
    LgbDataset,
    LgbModelConfig,
    LgbTrainConfig,
)
from numpy.testing import assert_allclose
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_pandas(titanic_data: tuple[pd.DataFrame, pd.Series[int]]) -> None:
    features, target = titanic_data
    dataset = LgbDataset(dataset=lgb.Dataset(features, label=target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={"objective": "binary", "seed": 1, "deterministic": True},
            callbacks=[lgb.early_stopping(100, verbose=False)],
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset, n_jobs=4)
    y_pred = model.predict(dataset, n_jobs=4)
    metric = accuracy_score(target, y_pred > 0.5)

    assert_allclose(metric, 0.81, atol=2.5e-02)
