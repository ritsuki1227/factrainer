from factrainer.core.cv.config import BaseTrainConfig


class Foo(BaseTrainConfig):
    foo: str


def test_cv() -> None:
    Foo("foo")
    # data = fetch_california_housing()
    # dataset = LgbDataset(dataset=lgb.Dataset(data.data, label=data.target))
    # config = LgbModelConfig.create(
    #     train_config=LgbTrainConfig(params={"objective": "regression"})
    # )
    # k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    # model = CvMlModel(config, k_fold)
    # model.train(dataset)
    # y_pred = model.predict(dataset)
    # r2_score(data.target, y_pred)
