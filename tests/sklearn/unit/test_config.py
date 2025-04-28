from unittest.mock import MagicMock, patch

import numpy as np
from factrainer.sklearn.config import (
    SklearnLearner,
    SklearnPredictConfig,
    SklearnPredictor,
    SklearnTrainConfig,
)
from factrainer.sklearn.dataset import SklearnDataset
from factrainer.sklearn.raw_model import SklearnModel
from sklearn.linear_model import LinearRegression


class TestSklearnLearner:
    def test_train(self) -> None:
        dataset = SklearnDataset(X=np.array([[1, 2], [3, 4.5]]), y=np.array([0, 1]))
        config = SklearnTrainConfig(estimator=LinearRegression())

        actual = SklearnLearner().train(dataset, None, config)

        assert hasattr(actual.estimator, "intercept_")
        assert hasattr(actual.estimator, "coef_")


class TestSklearnPredictor:
    @patch("sklearn.linear_model.LinearRegression.predict")
    def test_predict_with_default_config_regression(self, predict: MagicMock) -> None:
        estimator = LinearRegression()
        X, y = np.array([[1, 2], [3, 4.5]]), np.array([0, 1])
        dataset = SklearnDataset(X=X, y=y)
        model = SklearnModel(estimator=estimator)
        config = SklearnPredictConfig()
        expected = predict.return_value

        actual = SklearnPredictor().predict(dataset, model, config)

        assert actual == expected
        predict.assert_called_once_with(X)
