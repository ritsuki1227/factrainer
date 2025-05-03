from unittest.mock import MagicMock, PropertyMock, patch

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
