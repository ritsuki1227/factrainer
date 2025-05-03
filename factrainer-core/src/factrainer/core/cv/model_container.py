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
        self._learner = model_config.learner
        self._predictor = model_config.predictor
        self._train_config = model_config.train_config
        self._pred_config = model_config.pred_config
        self._k_fold = k_fold

    def train(self, train_dataset: T, n_jobs: int | None = None) -> None:
        datasets = SplittedDatasets.create(train_dataset, self._k_fold)
        self._cv_indices = datasets.indices
        self._model = CvLearner(self._learner).train(
            datasets.train, datasets.val, self.train_config, n_jobs
        )

    def predict(
        self,
        pred_dataset: T,
        n_jobs: int | None = None,
        mode: PredMode = PredMode.OOF_PRED,
    ) -> Prediction:
        match mode:
            case PredMode.OOF_PRED:
                datasets = IndexedDatasets[T].create(pred_dataset, self.cv_indices.test)
                return OutOfFoldPredictor(self._predictor).predict(
                    datasets, self.raw_model, self.pred_config, n_jobs
                )
            case PredMode.AVG_ENSEMBLE:
                return AverageEnsemblePredictor(self._predictor).predict(
                    pred_dataset, self.raw_model, self.pred_config, n_jobs
                )
            case _:
                raise ValueError(f"Invalid prediction mode: {mode}")

    @property
    def raw_model(self) -> RawModels[U]:
        return self._model

    @property
    def train_config(self) -> V:
        return self._train_config

    @train_config.setter
    def train_config(self, config: V) -> None:
        self._train_config = config

    @property
    def pred_config(self) -> W:
        return self._pred_config

    @pred_config.setter
    def pred_config(self, config: W) -> None:
        self._pred_config = config

    @property
    def cv_indices(self) -> SplittedDatasetsIndices:
        return self._cv_indices

    @property
    def k_fold(self) -> _BaseKFold | SplittedDatasetsIndices:
        return self._k_fold
