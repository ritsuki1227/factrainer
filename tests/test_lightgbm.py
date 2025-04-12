import lightgbm as lgb
import pytest
from factrainer.core import CvMlModel
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils._bunch import Bunch


@pytest.fixture
def _california_housing_data() -> Bunch:
    return fetch_california_housing()


def test_cv_model(_california_housing_data: Bunch) -> None:
    # data = fetch_california_housing()
    # dataset = LgbDataset(dataset=lgb.Dataset(data.data, label=data.target))
    dataset = LgbDataset(
        dataset=lgb.Dataset(
            _california_housing_data.data, label=_california_housing_data.target
        )
    )
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(params={"objective": "regression"})
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    model = CvMlModel(config, k_fold)

    model.train(dataset)
    y_pred = model.predict(dataset)
    r2_score(_california_housing_data.target, y_pred)


def test_single_model() -> None: ...
