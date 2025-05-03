from unittest.mock import MagicMock, PropertyMock, patch

from factrainer.base.config import (
    BaseLearner,
    BaseMlModelConfig,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
)
from factrainer.base.dataset import IndexableDataset
from factrainer.base.raw_model import RawModel
from factrainer.core.cv.config import PredMode
from factrainer.core.cv.dataset import SplittedDatasetsIndices
from factrainer.core.cv.model_container import CvModelContainer
from sklearn.model_selection._split import _BaseKFold


@patch("factrainer.core.cv.model_container.CvLearner", spec=BaseLearner)
@patch("factrainer.core.cv.dataset.SplittedDatasets.create")
class TestTrain:
    def test_train(
        self,
        create: MagicMock,
        CvLearner: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=IndexableDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        k_fold = MagicMock(spec=_BaseKFold | SplittedDatasetsIndices).return_value
        sut = CvModelContainer[
            IndexableDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config, k_fold)

        sut.train(dataset)

        create.assert_called_once_with(dataset, k_fold)
        CvLearner.assert_called_once_with(config.learner)
        CvLearner.return_value.train.assert_called_once_with(
            create.return_value.train,
            create.return_value.val,
            config.train_config,
            None,
        )
        assert sut.cv_indices == create.return_value.indices
        assert sut.raw_model == CvLearner.return_value.train.return_value

    def test_parallel(
        self,
        create: MagicMock,
        CvLearner: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=IndexableDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        k_fold = MagicMock(spec=_BaseKFold | SplittedDatasetsIndices).return_value
        sut = CvModelContainer[
            IndexableDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config, k_fold)

        sut.train(dataset, n_jobs=2)

        create.assert_called_once_with(dataset, k_fold)
        CvLearner.assert_called_once_with(config.learner)
        CvLearner.return_value.train.assert_called_once_with(
            create.return_value.train,
            create.return_value.val,
            config.train_config,
            2,
        )
        assert sut.cv_indices == create.return_value.indices
        assert sut.raw_model == CvLearner.return_value.train.return_value


@patch.object(
    CvModelContainer,
    "raw_model",
    new_callable=PropertyMock,
)
class TestPredict:
    @patch.object(
        CvModelContainer,
        "cv_indices",
        new_callable=PropertyMock,
    )
    @patch("factrainer.core.cv.dataset.IndexedDatasets.create")
    @patch("factrainer.core.cv.model_container.OutOfFoldPredictor", spec=BasePredictor)
    def test_oof_predict(
        self,
        MockOofPredictor: MagicMock,
        create: MagicMock,
        cv_indices: MagicMock,
        raw_model: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=IndexableDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        k_fold = MagicMock(spec=_BaseKFold | SplittedDatasetsIndices).return_value
        sut = CvModelContainer[
            IndexableDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config, k_fold)

        actual = sut.predict(dataset)

        create.assert_called_once_with(dataset, cv_indices.return_value.test)
        MockOofPredictor.assert_called_once_with(config.predictor)
        MockOofPredictor.return_value.predict.assert_called_once_with(
            create.return_value,
            raw_model.return_value,
            config.pred_config,
            None,
        )
        assert actual == MockOofPredictor.return_value.predict.return_value

    @patch(
        "factrainer.core.cv.model_container.AverageEnsemblePredictor",
        spec=BasePredictor,
    )
    def test_average_ensemble_predict(
        self,
        MockAvgEnsemblePredictor: MagicMock,
        raw_model: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=IndexableDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        k_fold = MagicMock(spec=_BaseKFold | SplittedDatasetsIndices).return_value
        sut = CvModelContainer[
            IndexableDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config, k_fold)

        actual = sut.predict(dataset, mode=PredMode.AVG_ENSEMBLE)

        MockAvgEnsemblePredictor.assert_called_once_with(config.predictor)
        MockAvgEnsemblePredictor.return_value.predict.assert_called_once_with(
            dataset,
            raw_model.return_value,
            config.pred_config,
            None,
        )
        assert actual == MockAvgEnsemblePredictor.return_value.predict.return_value

    @patch.object(
        CvModelContainer,
        "cv_indices",
        new_callable=PropertyMock,
    )
    @patch("factrainer.core.cv.dataset.IndexedDatasets.create")
    @patch("factrainer.core.cv.model_container.OutOfFoldPredictor", spec=BasePredictor)
    def test_parallel(
        self,
        MockOofPredictor: MagicMock,
        create: MagicMock,
        cv_indices: MagicMock,
        raw_model: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=IndexableDataset).return_value
        config = MagicMock(spec=BaseMlModelConfig).return_value
        k_fold = MagicMock(spec=_BaseKFold | SplittedDatasetsIndices).return_value
        sut = CvModelContainer[
            IndexableDataset, RawModel, BaseTrainConfig, BasePredictConfig
        ](config, k_fold)

        actual = sut.predict(dataset, 2)

        create.assert_called_once_with(dataset, cv_indices.return_value.test)
        MockOofPredictor.assert_called_once_with(config.predictor)
        MockOofPredictor.return_value.predict.assert_called_once_with(
            create.return_value,
            raw_model.return_value,
            config.pred_config,
            2,
        )
        assert actual == MockOofPredictor.return_value.predict.return_value


class TestModelConfig:
    def test_set_train_config(self) -> None:
        train_config = MagicMock(spec=BaseTrainConfig).return_value
        model_config = MagicMock(spec=BaseMlModelConfig).return_value
        model_config.train_config = train_config
        new_train_config = MagicMock(spec=BaseTrainConfig).return_value
        sut = CvModelContainer[
            MagicMock, MagicMock, BaseTrainConfig, BasePredictConfig
        ](model_config, MagicMock())

        sut.train_config = new_train_config

        assert sut.train_config == new_train_config
        assert model_config.train_config != new_train_config

    def test_set_pred_config(self) -> None:
        pred_config = MagicMock(spec=BasePredictConfig).return_value
        model_config = MagicMock(spec=BaseMlModelConfig).return_value
        model_config.pred_config = pred_config
        new_pred_config = MagicMock(spec=BasePredictConfig).return_value
        sut = CvModelContainer[
            MagicMock, MagicMock, BaseTrainConfig, BasePredictConfig
        ](model_config, MagicMock())

        sut.pred_config = new_pred_config

        assert sut.pred_config == new_pred_config
        assert model_config.pred_config != new_pred_config
