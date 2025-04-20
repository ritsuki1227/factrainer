from typing import Any
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
from factrainer.lightgbm.config import (
    LgbLearner,
    LgbModelConfig,
    LgbPredictConfig,
    LgbPredictor,
    LgbTrainConfig,
)
from factrainer.lightgbm.dataset.dataset import LgbDataset
from factrainer.lightgbm.raw_model import LgbModel
from lightgbm.basic import _LGBM_TrainDataType
from numpy import typing as npt


@patch("factrainer.lightgbm.config.LgbPredictor", spec=LgbPredictor)
@patch("factrainer.lightgbm.config.LgbLearner", spec=LgbLearner)
class TestCreateLgbModelConfig:
    def test_create_lgb_model_config_with_default(
        self, learner: MagicMock, predictor: MagicMock
    ) -> None:
        train_config = MagicMock(spec=LgbTrainConfig)
        expected = LgbModelConfig(
            learner=learner.return_value,
            predictor=predictor.return_value,
            train_config=train_config,
            pred_config=None,
        )
        actual = LgbModelConfig.create(train_config)
        assert actual == expected

    def test_create_lgb_model_config(
        self, learner: MagicMock, predictor: MagicMock
    ) -> None:
        train_config = MagicMock(spec=LgbTrainConfig)
        pred_config = MagicMock(spec=LgbPredictConfig)
        expected = LgbModelConfig(
            learner=learner.return_value,
            predictor=predictor.return_value,
            train_config=train_config,
            pred_config=pred_config,
        )
        actual = LgbModelConfig.create(train_config, pred_config)
        assert actual == expected


@patch("lightgbm.train")
class TestLgbLearner:
    def test_train_with_val(self, lgb_train: MagicMock) -> None:
        train_lgb_dataset = MagicMock(spec=lgb.Dataset)
        train_dataset = LgbDataset(dataset=train_lgb_dataset)
        val_lgb_dataset = MagicMock(spec=lgb.Dataset)
        val_dataset = LgbDataset(dataset=val_lgb_dataset)
        config = LgbTrainConfig(params={"foo": "bar"})
        lgb_train.return_value = MagicMock(spec=lgb.Booster)
        expected = LgbModel(model=lgb_train.return_value)
        sut = LgbLearner()

        actual = sut.train(train_dataset, val_dataset, config)

        assert actual == expected
        lgb_train.assert_called_once_with(
            params={"foo": "bar"},
            num_boost_round=100,
            valid_names=None,
            feval=None,
            init_model=None,
            keep_training_booster=False,
            callbacks=None,
            train_set=train_lgb_dataset,
            valid_sets=[val_lgb_dataset],
        )

    def test_train_without_val(self, lgb_train: MagicMock) -> None:
        train_lgb_dataset = MagicMock(spec=lgb.Dataset)
        train_dataset = LgbDataset(dataset=train_lgb_dataset)
        val_dataset = None
        config = LgbTrainConfig(params={"foo": "bar"})

        lgb_train.return_value = MagicMock(spec=lgb.Booster)
        expected = LgbModel(model=lgb_train.return_value)
        sut = LgbLearner()

        actual = sut.train(train_dataset, val_dataset, config)

        assert actual == expected
        lgb_train.assert_called_once_with(
            params={"foo": "bar"},
            num_boost_round=100,
            valid_names=None,
            feval=None,
            init_model=None,
            keep_training_booster=False,
            callbacks=None,
            train_set=train_lgb_dataset,
            valid_sets=None,
        )

    def test_train_without_default(self, lgb_train: MagicMock) -> None:
        def custom_metric(
            preds: npt.NDArray[Any], eval_data: lgb.Dataset
        ) -> tuple[str, float, bool]:
            return "custom_metric", 0.0, True

        def custom_callback() -> None: ...

        init_model = MagicMock(spec=lgb.Booster)
        callbacks = [custom_callback]
        train_lgb_dataset = MagicMock(spec=lgb.Dataset)
        train_dataset = LgbDataset(dataset=train_lgb_dataset)
        val_dataset = None
        config = LgbTrainConfig(
            params={"foo": "bar"},
            num_boost_round=1000,
            valid_names=["foo", "bar"],
            feval=custom_metric,
            init_model=init_model,
            keep_training_booster=True,
            callbacks=callbacks,
        )

        lgb_train.return_value = MagicMock(spec=lgb.Booster)
        expected = LgbModel(model=lgb_train.return_value)
        sut = LgbLearner()

        actual = sut.train(train_dataset, val_dataset, config)

        assert actual == expected
        lgb_train.assert_called_once_with(
            params={"foo": "bar"},
            num_boost_round=1000,
            valid_names=["foo", "bar"],
            feval=custom_metric,
            init_model=init_model,
            keep_training_booster=True,
            callbacks=callbacks,
            train_set=train_lgb_dataset,
            valid_sets=None,
        )


class TestLgbPredictor:
    def test_predict_without_config(self) -> None:
        data = MagicMock(spec=_LGBM_TrainDataType)
        dataset = LgbDataset(dataset=lgb.Dataset(data))
        raw_model = MagicMock(spec=lgb.Booster)
        raw_model.predict.return_value = MagicMock(spec=np.ndarray)
        model = LgbModel(model=raw_model)
        sut = LgbPredictor()

        actual = sut.predict(dataset, model, None)

        assert actual == raw_model.predict.return_value
        raw_model.predict.assert_called_once_with(data)

    def test_predict_with_config(self) -> None:
        data = MagicMock(spec=_LGBM_TrainDataType)
        dataset = LgbDataset(dataset=lgb.Dataset(data))
        raw_model = MagicMock(spec=lgb.Booster)
        raw_model.predict.return_value = MagicMock(spec=np.ndarray)
        model = LgbModel(model=raw_model)
        config = LgbPredictConfig(
            start_iteration=10,
            num_iteration=1000,
            raw_score=True,
            pred_leaf=True,
            pred_contrib=True,
            data_has_header=True,
            validate_features=True,
        )
        sut = LgbPredictor()

        actual = sut.predict(dataset, model, config)

        assert actual == raw_model.predict.return_value
        raw_model.predict.assert_called_once_with(
            data=data,
            start_iteration=10,
            num_iteration=1000,
            raw_score=True,
            pred_leaf=True,
            pred_contrib=True,
            data_has_header=True,
            validate_features=True,
        )

    def test_predict_with_custom_kwargs_config(self) -> None:
        data = MagicMock(spec=_LGBM_TrainDataType)
        dataset = LgbDataset(dataset=lgb.Dataset(data))
        raw_model = MagicMock(spec=lgb.Booster)
        raw_model.predict.return_value = MagicMock(spec=np.ndarray)
        model = LgbModel(model=raw_model)
        config = LgbPredictConfig(
            start_iteration=10,
            num_iteration=1000,
            raw_score=True,
            pred_leaf=True,
            pred_contrib=True,
            data_has_header=True,
            validate_features=True,
            custom_kwargs="foo",
        )
        sut = LgbPredictor()

        actual = sut.predict(dataset, model, config)

        assert actual == raw_model.predict.return_value
        raw_model.predict.assert_called_once_with(
            data=data,
            start_iteration=10,
            num_iteration=1000,
            raw_score=True,
            pred_leaf=True,
            pred_contrib=True,
            data_has_header=True,
            validate_features=True,
            custom_kwargs="foo",
        )
