import numpy as np
from factrainer.sklearn.config import SklearnLearner, SklearnTrainConfig
from factrainer.sklearn.dataset import SklearnDataset
from sklearn.linear_model import LinearRegression


class TestSklearnLearner:
    def test_train(self) -> None:
        dataset = SklearnDataset(X=np.array([[1, 2], [3, 4.5]]), y=np.array([0, 1]))
        config = SklearnTrainConfig(estimator=LinearRegression())

        actual = SklearnLearner().train(dataset, None, config)

        assert hasattr(actual.estimator, "intercept_")
        assert hasattr(actual.estimator, "coef_")
