from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from factrainer.base.config import BaseMlModelConfig, BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import BaseDataset
from factrainer.base.raw_model import RawModel
from factrainer.core.single import SingleModelContainer


class TestTrain:
    def test_without_val(self) -> None:
        train_dataset = MagicMock(spec=BaseDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        sut = SingleModelContainer[
            BaseDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config)

        sut.train(train_dataset)

        assert sut.raw_model == config.learner.train.return_value
        config.learner.train.assert_called_once_with(
            train_dataset, None, config.train_config
        )

    def test_with_val(self) -> None:
        train_dataset = MagicMock(spec=BaseDataset).return_value
        val_dataset = MagicMock(spec=BaseDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        sut = SingleModelContainer[
            BaseDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config)

        sut.train(train_dataset, val_dataset)

        assert sut.raw_model == config.learner.train.return_value
        config.learner.train.assert_called_once_with(
            train_dataset, val_dataset, config.train_config
        )


@patch.object(
    SingleModelContainer,
    "raw_model",
    new_callable=PropertyMock,
)
def test_predict(raw_model: MagicMock) -> None:
    pred_dataset = MagicMock(spec=BaseDataset).return_value
    config = MagicMock(spec=BaseMlModelConfig).return_value
    expected = config.predictor.predict.return_value
    sut = SingleModelContainer[
        BaseDataset, RawModel, BaseTrainConfig, BasePredictConfig
    ](config)

    actual = sut.predict(pred_dataset)

    assert actual == expected
    config.predictor.predict.assert_called_once_with(
        pred_dataset, raw_model.return_value, config.pred_config
    )


class TestModelConfig:
    def test_set_train_config(self) -> None:
        train_config = MagicMock(spec=BaseTrainConfig).return_value
        model_config = MagicMock(spec=BaseMlModelConfig).return_value
        model_config.train_config = train_config
        new_train_config = MagicMock(spec=BaseTrainConfig).return_value
        sut = SingleModelContainer[
            MagicMock, MagicMock, BaseTrainConfig, BasePredictConfig
        ](model_config)

        sut.train_config = new_train_config

        assert sut.train_config == new_train_config
        assert model_config.train_config != new_train_config

    def test_set_pred_config(self) -> None:
        pred_config = MagicMock(spec=BasePredictConfig).return_value
        model_config = MagicMock(spec=BaseMlModelConfig).return_value
        model_config.pred_config = pred_config
        new_pred_config = MagicMock(spec=BasePredictConfig).return_value
        sut = SingleModelContainer[
            MagicMock, MagicMock, BaseTrainConfig, BasePredictConfig
        ](model_config)

        sut.pred_config = new_pred_config

        assert sut.pred_config == new_pred_config
        assert model_config.pred_config != new_pred_config


class TestEvaluate:
    def test_evaluate(self) -> None:
        config = MagicMock(spec=BaseMlModelConfig).return_value
        sut = SingleModelContainer[
            BaseDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config)

        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2])
        eval_func = MagicMock()
        eval_func.return_value = 0.95

        result = sut.evaluate(y_true, y_pred, eval_func)

        eval_func.assert_called_once_with(y_true, y_pred)
        assert result == 0.95

    def test_invalid_input_types(self) -> None:
        config = MagicMock(spec=BaseMlModelConfig).return_value
        sut = SingleModelContainer[
            BaseDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config)

        y_true_list = [1, 2, 3, 4]
        y_pred_array = np.array([1.1, 2.1, 2.9, 4.2])
        eval_func = MagicMock()

        with pytest.raises(ValueError):
            sut.evaluate(y_true_list, y_pred_array, eval_func)  # type: ignore[arg-type]

        y_true_array = np.array([1, 2, 3, 4])
        y_pred_list = [1.1, 2.1, 2.9, 4.2]

        with pytest.raises(ValueError):
            sut.evaluate(y_true_array, y_pred_list, eval_func)  # type: ignore[arg-type]
