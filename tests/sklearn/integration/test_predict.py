from __future__ import annotations

import tempfile
from typing import Any

import joblib
import numpy as np
import pytest
from factrainer.core import CvModelContainer, PredMode
from factrainer.sklearn import (
    SklearnDataset,
    SklearnModel,
    SklearnModelConfig,
    SklearnPredictConfig,
    SklearnPredictMethod,
    SklearnTrainConfig,
)
from numpy import typing as npt
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TestClassification:
    @pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
    def test_preidct_auto(
        self,
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

    def test_predict(
        self,
        breast_cancer_data: tuple[
            npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]
        ],
    ) -> None:
        features, target = breast_cancer_data
        dataset = SklearnDataset(X=features, y=target)
        config = SklearnModelConfig.create(
            train_config=SklearnTrainConfig(estimator=SVC(kernel="linear")),
            pred_config=SklearnPredictConfig(
                predict_method=SklearnPredictMethod.PREDICT
            ),
        )
        k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
        model = CvModelContainer(config, k_fold)
        model.train(dataset)
        y_pred = model.predict(dataset)
        metric = accuracy_score(target, y_pred > 0)

        assert_allclose(metric, 0.95, atol=2.5e-02)

    @pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
    def test_preidct_proba(
        self,
        iris_data: tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]],
    ) -> None:
        features, target = iris_data
        dataset = SklearnDataset(X=features, y=target)
        config = SklearnModelConfig.create(
            train_config=SklearnTrainConfig(estimator=LogisticRegression()),
            pred_config=SklearnPredictConfig(
                predict_method=SklearnPredictMethod.PREDICT_PROBA
            ),
        )
        k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
        model = CvModelContainer(config, k_fold)
        model.train(dataset)
        y_pred = model.predict(dataset)
        y_pred = np.argmax(y_pred, axis=1)
        metric = f1_score(target, y_pred, average="micro")

        assert_allclose(metric, 0.95, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_set_pred_config_after_trainig(
    breast_cancer_data: tuple[npt.NDArray[np.number[Any]], npt.NDArray[np.number[Any]]],
) -> None:
    features, target = breast_cancer_data
    dataset = SklearnDataset(X=features, y=target)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=Pipeline(
                [("scaler", StandardScaler()), ("lr", LogisticRegression())]
            )
        ),
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(dataset)
    y_pred_proba = model.predict(dataset)
    model.pred_config = SklearnPredictConfig(
        predict_method=SklearnPredictMethod.PREDICT
    )
    y_pred = model.predict(dataset)

    assert y_pred_proba.ndim == 2
    assert y_pred.ndim == 1


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
    train_dataset = SklearnDataset(X=train_X, y=train_y)
    test_dataset = SklearnDataset(X=test_X, y=test_y)
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=RandomForestRegressor(random_state=100, n_jobs=-1)
        ),
    )
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    model = CvModelContainer(config, k_fold)
    model.train(train_dataset)
    y_pred = model.predict(test_dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
    metric = r2_score(test_y, y_pred)

    assert_allclose(metric, 0.8, atol=2.5e-02)


@pytest.mark.flaky(reruns=3, reruns_delay=5, only_rerun=["HTTPError"])
def test_cv_model_picklable(
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
    with tempfile.NamedTemporaryFile() as fp:
        joblib.dump(model, fp.name)
        loaded_model: CvModelContainer[
            SklearnDataset, SklearnModel, SklearnTrainConfig, SklearnPredictConfig
        ] = joblib.load(fp.name)
    y_pred = loaded_model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=1)
    metric = f1_score(target, y_pred, average="micro")

    assert_allclose(metric, 0.95, atol=2.5e-02)
