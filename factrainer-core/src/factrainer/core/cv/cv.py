from factrainer.base.config import BaseMlModelConfig, BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import IndexableDataset, Prediction
from factrainer.base.raw_model import RawModel
from sklearn.model_selection._split import _BaseKFold

from ..trait import BaseModelContainer
from .config import CvMlModelConfig
from .dataset import IndexedDatasets, SplittedDatasets, SplittedDatasetsIndices
from .raw_model import CvRawModels


class CvModelContainer[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModelContainer[T, CvRawModels[U], V, W]):
    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
        k_fold: _BaseKFold | SplittedDatasetsIndices,
    ) -> None:
        self._model_config = CvMlModelConfig.from_config(model_config)
        self._k_fold = k_fold

    def train(self, train_dataset: T, n_jobs: int | None = None) -> None:
        datasets = SplittedDatasets.create(train_dataset, self._k_fold)
        self._cv_indices = datasets.indices
        self._model = self._model_config.learner.train(
            datasets.train, datasets.val, self._model_config.train_config, n_jobs
        )

    def predict(self, pred_dataset: T, n_jobs: int | None = None) -> Prediction:
        datasets = IndexedDatasets.create(pred_dataset, self.cv_indices.test)
        return self._model_config.predictor.predict(
            datasets, self.raw_model, self._model_config.pred_config, n_jobs
        )

    @property
    def raw_model(self) -> CvRawModels[U]:
        return self._model

    @property
    def train_config(self) -> V:
        return self._model_config.train_config

    @train_config.setter
    def train_config(self, config: V) -> None:
        self._model_config.train_config = config

    @property
    def pred_config(self) -> W | None:
        return self._model_config.pred_config

    @pred_config.setter
    def pred_config(self, config: W | None) -> None:
        self._model_config.pred_config = config

    @property
    def cv_indices(self) -> SplittedDatasetsIndices:
        return self._cv_indices

    @property
    def k_fold(self) -> _BaseKFold | SplittedDatasetsIndices:
        return self._k_fold
