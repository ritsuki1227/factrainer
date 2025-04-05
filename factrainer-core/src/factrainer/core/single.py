from factrainer.base.config import (
    BaseMlModelConfig,
    BasePredictConfig,
    BaseTrainConfig,
)
from factrainer.base.dataset import BaseDataset, Prediction
from factrainer.base.raw_model import RawModel

from .trait import (
    PredictorTrait,
    ValidatableTrainerTrait,
)


class SingleMlModel[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](ValidatableTrainerTrait[T, V], PredictorTrait[T, U, W]):
    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
    ) -> None:
        self.model_config = model_config

    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        self._model = self.model_config.learner.train(
            train_dataset, val_dataset, self.model_config.train_config
        )

    def predict(self, dataset: T) -> Prediction:
        return self.model_config.predictor.predict(
            dataset, self.raw_model, self.model_config.pred_config
        )

    @property
    def raw_model(self) -> U:
        return self._model

    @property
    def train_config(self) -> V:
        return self.model_config.train_config

    @train_config.setter
    def train_config(self, config: V) -> None:
        self.model_config.train_config = config

    @property
    def pred_config(self) -> W | None:
        return self.model_config.pred_config

    @pred_config.setter
    def pred_config(self, config: W | None) -> None:
        self.model_config.pred_config = config
