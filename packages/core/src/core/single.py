from base.config import (
    BaseMlModelConfig,
    BasePredictConfig,
    BaseTrainConfig,
    Prediction,
)
from base.dataset import Dataset
from base.raw_model import RawModel

from .trait import (
    PredictorTrait,
    TrainValDataset,
    ValidatableTrainerTrait,
)


class SingleMlModel[T: Dataset, U: RawModel, V: BaseTrainConfig, W: BasePredictConfig](
    ValidatableTrainerTrait[T], PredictorTrait[T, U]
):
    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
    ) -> None:
        self.model_config = model_config

    def train(self, dataset: TrainValDataset[T]) -> None:
        self._model = self.model_config.learner.train(
            dataset.train, dataset.val, self.model_config.train_config
        )

    def predict(self, dataset: T) -> Prediction:
        return self.model_config.predictor.predict(
            dataset, self.model, self.model_config.pred_config
        )

    @property
    def model(self) -> U:
        return self._model
