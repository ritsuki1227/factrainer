from factrainer.base.config import BaseMlModelConfig, BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import IndexableDataset, Prediction
from factrainer.base.raw_model import RawModel
from sklearn.model_selection._split import _BaseKFold

from ..model_container import BaseModelContainer
from .config import AverageEnsemblePredictor, CvLearner, OutOfFoldPredictor, PredMode
from .dataset import IndexedDatasets, SplittedDatasets, SplittedDatasetsIndices
from .raw_model import RawModels


class CvModelContainer[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModelContainer[T, RawModels[U], V, W]):
    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
        k_fold: _BaseKFold | SplittedDatasetsIndices,
    ) -> None:
        self._model_config = model_config
        self._k_fold = k_fold

    def train(self, train_dataset: T, n_jobs: int | None = None) -> None:
        datasets = SplittedDatasets.create(train_dataset, self._k_fold)
        self._cv_indices = datasets.indices
        self._model = CvLearner(self._model_config.learner).train(
            datasets.train, datasets.val, self._model_config.train_config, n_jobs
        )

    def predict(
        self,
        pred_dataset: T,
        n_jobs: int | None = None,
        mode: PredMode = PredMode.OOF_PRED,
    ) -> Prediction:
        if mode == PredMode.AVG_ENSEMBLE:
            return AverageEnsemblePredictor(self._model_config.predictor).predict(
                pred_dataset, self.raw_model, self._model_config.pred_config, n_jobs
            )
        elif mode == PredMode.OOF_PRED:
            datasets = IndexedDatasets[T].create(pred_dataset, self.cv_indices.test)
            return OutOfFoldPredictor(self._model_config.predictor).predict(
                datasets, self.raw_model, self._model_config.pred_config, n_jobs
            )
        else:
            raise ValueError

    @property
    def raw_model(self) -> RawModels[U]:
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
