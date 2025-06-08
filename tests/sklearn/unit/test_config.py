from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
from factrainer.sklearn.config import (
    SklearnLearner,
    SklearnModelConfig,
    SklearnPredictConfig,
    SklearnPredictMethod,
    SklearnPredictor,
    SklearnTrainConfig,
)
from factrainer.sklearn.dataset.dataset import SklearnDataset
from factrainer.sklearn.raw_model import SklearnModel
from sklearn.linear_model import LinearRegression, LogisticRegression


class TestSklearnTrainConfig:
    def test_normal(self) -> None:
        estimator = LinearRegression()
        config = SklearnTrainConfig(estimator=estimator)

        assert config.estimator is estimator  # type: ignore[comparison-overlap]

    def test_with_additional_kwargs(self) -> None:
        estimator = LogisticRegression()
        sample_weight = np.array([1.0, 2.0, 1.5])
        config = SklearnTrainConfig(estimator=estimator, sample_weight=sample_weight)

        assert config.estimator is estimator  # type: ignore[comparison-overlap]
        assert config.model_extra is not None
        assert np.array_equal(config.model_extra["sample_weight"], sample_weight)


class TestSklearnPredictConfig:
    def test_normal(self) -> None:
        config = SklearnPredictConfig()

        assert config.predict_method == SklearnPredictMethod.AUTO

    def test_predict_method(self) -> None:
        config = SklearnPredictConfig(predict_method=SklearnPredictMethod.PREDICT)

        assert config.predict_method == SklearnPredictMethod.PREDICT

    def test_with_additional_kwargs(self) -> None:
        config = SklearnPredictConfig(
            predict_method=SklearnPredictMethod.PREDICT, batch_size=32
        )

        assert config.predict_method == SklearnPredictMethod.PREDICT
        assert config.model_extra is not None
        assert config.model_extra["batch_size"] == 32


@patch("factrainer.sklearn.config.SklearnPredictor", spec=SklearnPredictor)
@patch("factrainer.sklearn.config.SklearnLearner", spec=SklearnLearner)
class TestSklearnModelConfig:
    @patch("factrainer.sklearn.config.SklearnPredictConfig", spec=SklearnPredictConfig)
    def test_create_sklearn_model_config_with_default(
        self,
        pred_config: MagicMock,
        learner: MagicMock,
        predictor: MagicMock,
    ) -> None:
        train_config = MagicMock(spec=SklearnTrainConfig)
        expected = SklearnModelConfig(
            learner=learner.return_value,
            predictor=predictor.return_value,
            train_config=train_config,
            pred_config=pred_config.return_value,
        )
        actual = SklearnModelConfig.create(train_config)
        assert actual == expected

    def test_create_sklearn_model_config(
        self, learner: MagicMock, predictor: MagicMock
    ) -> None:
        train_config = MagicMock(spec=SklearnTrainConfig)
        pred_config = MagicMock(spec=SklearnPredictConfig)
        expected = SklearnModelConfig(
            learner=learner.return_value,
            predictor=predictor.return_value,
            train_config=train_config,
            pred_config=pred_config,
        )
        actual = SklearnModelConfig.create(train_config, pred_config)
        assert actual == expected


class TestSklearnLearner:
    def test_train(self) -> None:
        dataset = SklearnDataset(X=np.array([[1, 2], [3, 4.5]]), y=np.array([0, 1]))
        config = SklearnTrainConfig(estimator=LinearRegression())

        actual = SklearnLearner().train(dataset, None, config)

        assert hasattr(actual.estimator, "intercept_")
        assert hasattr(actual.estimator, "coef_")

    def test_pandas(self) -> None:
        dataset = SklearnDataset(
            X=pd.DataFrame([[1, 2], [3, 4.5]]), y=pd.Series([0, 1])
        )
        config = SklearnTrainConfig(estimator=LinearRegression())

        actual = SklearnLearner().train(dataset, None, config)

        assert hasattr(actual.estimator, "intercept_")
        assert hasattr(actual.estimator, "coef_")

    def test_polars(self) -> None:
        dataset = SklearnDataset(
            X=pl.DataFrame({"a": [1, 2], "b": [3.0, 4.5]}), y=pl.Series([0, 1])
        )
        config = SklearnTrainConfig(estimator=LinearRegression())

        actual = SklearnLearner().train(dataset, None, config)

        assert hasattr(actual.estimator, "intercept_")
        assert hasattr(actual.estimator, "coef_")


class TestSklearnPredictor:
    @patch("sklearn.linear_model.LinearRegression.predict")
    def test_regression(self, predict: MagicMock) -> None:
        estimator = LinearRegression()
        X, y = np.array([[1, 2], [3, 4.5]]), np.array([0, 1])
        dataset = SklearnDataset(X=X, y=y)
        model = SklearnModel(estimator=estimator)
        config = SklearnPredictConfig()
        expected = predict.return_value

        actual = SklearnPredictor().predict(dataset, model, config)

        assert actual == expected
        predict.assert_called_once_with(X)

    class TestClassification:
        @patch("sklearn.linear_model.LogisticRegression.predict_proba")
        def test_auto(self, predict_proba: MagicMock) -> None:
            estimator = LogisticRegression()
            X, y = np.array([[1, 2], [3, 4.5]]), np.array([0, 1])
            dataset = SklearnDataset(X=X, y=y)
            model = SklearnModel(estimator=estimator)
            config = SklearnPredictConfig()
            expected = predict_proba.return_value

            actual = SklearnPredictor().predict(dataset, model, config)

            assert actual == expected
            predict_proba.assert_called_once_with(X)

        @patch("sklearn.linear_model.LogisticRegression.predict")
        def test_predict(self, predict: MagicMock) -> None:
            estimator = LogisticRegression()
            X, y = np.array([[1, 2], [3, 4.5]]), np.array([0, 1])
            dataset = SklearnDataset(X=X, y=y)
            model = SklearnModel(estimator=estimator)
            config = SklearnPredictConfig(predict_method=SklearnPredictMethod.PREDICT)
            expected = predict.return_value

            actual = SklearnPredictor().predict(dataset, model, config)

            assert actual == expected
            predict.assert_called_once_with(X)

        @patch("sklearn.linear_model.LogisticRegression.predict_proba")
        def test_predict_proba(self, predict_proba: MagicMock) -> None:
            estimator = LogisticRegression()
            X, y = np.array([[1, 2], [3, 4.5]]), np.array([0, 1])
            dataset = SklearnDataset(X=X, y=y)
            model = SklearnModel(estimator=estimator)
            config = SklearnPredictConfig(
                predict_method=SklearnPredictMethod.PREDICT_PROBA
            )
            expected = predict_proba.return_value

            actual = SklearnPredictor().predict(dataset, model, config)

            assert actual == expected
            predict_proba.assert_called_once_with(X)
