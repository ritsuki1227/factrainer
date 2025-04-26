from typing import Self

from factrainer.base.config import (
    BaseLearner,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    MlModelConfig,
)
from factrainer.base.dataset import Prediction

from .dataset import SklearnDataset
from .raw_model import SklearnModel


class SklearnTrainConfig(BaseTrainConfig): ...


class SklearnPredictConfig(BasePredictConfig): ...


class SklearnLearner(BaseLearner[SklearnDataset, SklearnModel, SklearnTrainConfig]):
    def train(
        self,
        train_dataset: SklearnDataset,
        val_dataset: SklearnDataset | None,
        config: SklearnTrainConfig,
    ) -> SklearnModel:
        raise NotImplementedError


class SklearnPredictor(
    BasePredictor[SklearnDataset, SklearnModel, SklearnPredictConfig]
):
    def predict(
        self,
        dataset: SklearnDataset,
        raw_model: SklearnModel,
        config: SklearnPredictConfig | None,
    ) -> Prediction:
        raise NotImplementedError


class SklearnModelConfig(
    MlModelConfig[
        SklearnDataset, SklearnModel, SklearnTrainConfig, SklearnPredictConfig
    ]
):
    learner: SklearnLearner
    predictor: SklearnPredictor
    train_config: SklearnTrainConfig
    predict_config: SklearnPredictConfig

    @classmethod
    def create(
        cls,
        train_config: SklearnTrainConfig,
        pred_config: SklearnPredictConfig | None = None,
    ) -> Self:
        return cls(
            learner=SklearnLearner(),
            predictor=SklearnPredictor(),
            train_config=train_config,
            predict_config=pred_config,
        )
