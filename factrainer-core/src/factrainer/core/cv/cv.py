from factrainer.base.config import BaseMlModelConfig, BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import IndexableDataset, Prediction
from factrainer.base.raw_model import RawModel
from sklearn.model_selection._split import _BaseKFold

from ..single import SingleMlModel
from ..trait import PredictorTrait, TrainerTrait
from .config import CvMlModelConfig
from .dataset import IndexedDatasets, SplittedDatasets, SplittedDatasetsIndices
from .raw_model import CvRawModels


class CvMlModel[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](TrainerTrait[T, V], PredictorTrait[T, CvRawModels[U], W]):
    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
        k_fold: _BaseKFold,
        n_jobs_train: int | None = None,
        n_jobs_predict: int | None = None,
    ) -> None:
        self._cv_model = SingleMlModel(
            CvMlModelConfig.from_config(model_config, n_jobs_train, n_jobs_predict)
        )
        self._k_fold = k_fold

    def train(self, dataset: T) -> None:
        datasets = SplittedDatasets.create(dataset, self._k_fold)
        self._cv_indices = datasets.indices
        self._cv_model.train(datasets.train, datasets.test)

    def predict(self, dataset: T) -> Prediction:
        datasets = IndexedDatasets.create(dataset, self.cv_indices.test)
        return self._cv_model.predict(datasets)

    @property
    def raw_model(self) -> CvRawModels[U]:
        return self._cv_model.raw_model

    @property
    def cv_indices(self) -> SplittedDatasetsIndices:
        return self._cv_indices

    @property
    def n_jobs_train(self) -> int | None:
        return self._cv_model.model_config.n_jobs_train  # type: ignore

    @n_jobs_train.setter
    def n_jobs_train(self, n_jobs: int | None) -> None:
        self._cv_model.model_config.n_jobs_train = n_jobs  # type: ignore

    @property
    def n_jobs_predict(self) -> int | None:
        return self._cv_model.model_config.n_jobs_predict  # type: ignore

    @n_jobs_predict.setter
    def n_jobs_predict(self, n_jobs: int | None) -> None:
        self._cv_model.model_config.n_jobs_predict = n_jobs  # type: ignore

    @property
    def train_config(self) -> V:
        return self._cv_model.train_config

    @train_config.setter
    def train_config(self, config: V) -> None:
        self._cv_model.train_config = config

    @property
    def pred_config(self) -> W | None:
        return self._cv_model.pred_config

    @pred_config.setter
    def pred_config(self, config: W | None) -> None:
        self._cv_model.pred_config = config
