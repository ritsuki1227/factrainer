from enum import Enum, auto
from typing import Self

from factrainer.base.config import (
    BaseLearner,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    MlModelConfig,
)
from factrainer.base.dataset import Prediction
from pydantic import ConfigDict

from .dataset import SklearnDataset
from .raw_model import Predictable, ProbPredictable, SklearnModel


class SklearnTrainConfig(BaseTrainConfig):
    estimator: Predictable | ProbPredictable
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PredictMethod(Enum):
    PREDICT = auto()
    PREDICT_PROBA = auto()


class SklearnPredictConfig(BasePredictConfig):
    predict_method: PredictMethod = PredictMethod.PREDICT
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SklearnLearner(BaseLearner[SklearnDataset, SklearnModel, SklearnTrainConfig]):
    def train(
        self,
        train_dataset: SklearnDataset,
        val_dataset: SklearnDataset | None,
        config: SklearnTrainConfig,
    ) -> SklearnModel:
        config.estimator.fit(
            train_dataset.X,
            train_dataset.y,
            **(config.model_extra if config.model_extra is not None else {}),
        )
        return SklearnModel(estimator=config.estimator)


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
        if hasattr(raw_model.estimator, "predict_proba"):
            return raw_model.estimator.predict_proba(
                dataset.X,
                **dict(config if config is not None else {}),
            )
        elif hasattr(raw_model.estimator, "predict"):
            return raw_model.estimator.predict(
                dataset.X,
                **dict(config if config is not None else {}),
            )
        else:
            raise ValueError("The model is not a valid classifier or regressor.")
        # match raw_model.estimator:
        #     case ClassifierProtocol():
        #         return raw_model.estimator.predict_proba(
        #             dataset.X,
        #         )
        #     case RegressorProtocol():
        #         return raw_model.estimator.predict(dataset.X, **dict(config))
        #     case _:
        #         raise ValueError("The model is not a valid classifier or regressor.")


class SklearnModelConfig(
    MlModelConfig[
        SklearnDataset, SklearnModel, SklearnTrainConfig, SklearnPredictConfig
    ]
):
    learner: SklearnLearner
    predictor: SklearnPredictor
    train_config: SklearnTrainConfig
    predict_config: SklearnPredictConfig | None

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
