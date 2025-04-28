from copy import deepcopy
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
    AUTO = auto()
    PREDICT = auto()
    PREDICT_PROBA = auto()


class SklearnPredictConfig(BasePredictConfig):
    predict_method: PredictMethod = PredictMethod.AUTO
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SklearnLearner(BaseLearner[SklearnDataset, SklearnModel, SklearnTrainConfig]):
    def train(
        self,
        train_dataset: SklearnDataset,
        val_dataset: SklearnDataset | None,
        config: SklearnTrainConfig,
    ) -> SklearnModel:
        if train_dataset.y is None:
            raise ValueError("y cannot be None")
        _config = deepcopy(config)
        _config.estimator.fit(
            train_dataset.X,
            train_dataset.y,
            **(config.model_extra if config.model_extra is not None else {}),
        )
        return SklearnModel(estimator=_config.estimator)


class SklearnPredictor(
    BasePredictor[SklearnDataset, SklearnModel, SklearnPredictConfig]
):
    def predict(
        self,
        dataset: SklearnDataset,
        raw_model: SklearnModel,
        config: SklearnPredictConfig,
    ) -> Prediction:
        match config.predict_method:
            case PredictMethod.PREDICT_PROBA:
                raise NotImplementedError
            case PredictMethod.PREDICT:
                raise NotImplementedError
            case PredictMethod.AUTO:
                if hasattr(raw_model.estimator, "predict_proba"):
                    return raw_model.estimator.predict_proba(
                        dataset.X,
                        **(
                            config.model_extra if config.model_extra is not None else {}
                        ),
                    )
                elif hasattr(raw_model.estimator, "predict"):
                    return raw_model.estimator.predict(
                        dataset.X,
                        **(
                            config.model_extra if config.model_extra is not None else {}
                        ),
                    )
                else:
                    raise ValueError(
                        "The model is not a valid classifier or regressor."
                    )
            case _:
                raise ValueError("Invalid predict method")


class SklearnModelConfig(
    MlModelConfig[
        SklearnDataset, SklearnModel, SklearnTrainConfig, SklearnPredictConfig
    ]
):
    learner: SklearnLearner
    predictor: SklearnPredictor
    train_config: SklearnTrainConfig
    pred_config: SklearnPredictConfig

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
            pred_config=pred_config
            if pred_config is not None
            else SklearnPredictConfig(),
        )
