import lightgbm as lgb
import pytest
from factrainer.core import CvMlModel, SingleMlModel
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils._bunch import Bunch


@pytest.fixture
def _california_housing_data() -> Bunch:
    return fetch_california_housing()


def test_cv_model(_california_housing_data: Bunch) -> None:
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
    metric = r2_score(_california_housing_data.target, y_pred)

    assert (metric > 0.8) and (metric < 0.85)


def test_single_model(_california_housing_data: Bunch) -> None:
    train_X, test_X, train_y, test_y = train_test_split(
        _california_housing_data.data,
        _california_housing_data.target,
        test_size=0.2,
        random_state=1,
    )
    train_dataset = LgbDataset(dataset=lgb.Dataset(train_X, train_y))
    val_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    test_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(params={"objective": "regression"})
    )
    model = SingleMlModel(config)

    model.train(train_dataset, val_dataset)
    y_pred = model.predict(test_dataset)
    metric = r2_score(test_y, y_pred)

    assert (metric > 0.8) and (metric < 0.85)
