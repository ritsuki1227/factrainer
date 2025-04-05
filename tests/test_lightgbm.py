import lightgbm as lgb
from factrainer.core import CvMlModel
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def test_single_model() -> None:
    data = fetch_california_housing()
    dataset = LgbDataset(dataset=lgb.Dataset(data.data, label=data.target))
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(params={"objective": "regression"})
    )
    k_fold = KFold(n_splits=4, shuffle=True, random_state=1227)
    model = CvMlModel(config, k_fold)

    model.train(dataset)
    y_pred = model.predict(dataset)
    r2_score(data.target, y_pred)
