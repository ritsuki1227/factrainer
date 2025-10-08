from typing import Literal, Sequence, overload

import numpy as np
from factrainer.base.config import BaseMlModelConfig, BasePredictConfig, BaseTrainConfig
from factrainer.base.dataset import IndexableDataset, Prediction, Target
from factrainer.base.raw_model import RawModel
from sklearn.model_selection._split import _BaseKFold

from ..model_container import BaseModelContainer, EvalFunc
from .config import (
    AverageEnsemblePredictor,
    CvLearner,
    EvalMode,
    OutOfFoldPredictor,
    PredMode,
)
from .dataset import IndexedDatasets, SplittedDatasets, SplittedDatasetsIndices
from .raw_model import RawModels


class CvModelContainer[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModelContainer[T, RawModels[U], V, W]):
    """Cross-validation model container for machine learning models.

    This class provides a container for cross-validation models. It takes a model
    configuration and a cross-validation splitter, and provides methods for training
    models and making predictions.

    Parameters
    ----------
    model_config : BaseMlModelConfig
        The model configuration, which includes the learner, predictor, training
        configuration, and prediction configuration.
    k_fold : _BaseKFold or SplittedDatasetsIndices
        The cross-validation splitter, which can be either a scikit-learn _BaseKFold
        object or a SplittedDatasetsIndices object. If _BaseKFold is specified,
        the same indices will be used for both validation and test data. To specify
        custom indices without such constraints, use SplittedDatasetsIndices.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import r2_score
    >>> from sklearn.model_selection import KFold
    >>> from factrainer.core import CvModelContainer, EvalMode
    >>> from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
    >>>
    >>> # Load data
    >>> X, y = make_regression()
    >>> dataset = LgbDataset(dataset=lgb.Dataset(X, label=y))
    >>>
    >>> # Configure model
    >>> config = LgbModelConfig.create(
    ...     train_config=LgbTrainConfig(
    ...         params={"objective": "regression", "verbose": -1},
    ...         callbacks=[lgb.early_stopping(100, verbose=False)],
    ...     ),
    ... )
    >>>
    >>> # Set up cross-validation
    >>> k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
    >>>
    >>> # Create and train model
    >>> model = CvModelContainer(config, k_fold)
    >>> model.train(dataset, n_jobs=4)
    >>>
    >>> # Get OOF predictions
    >>> y_pred = model.predict(dataset, n_jobs=4)
    >>>
    >>> # Evaluate predictions
    >>> metric = model.evaluate(y, y_pred, r2_score)
    >>>
    >>> # Or get per-fold metrics
    >>> metrics = model.evaluate(y, y_pred, r2_score, eval_mode=EvalMode.FOLD_WISE)
    """

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
        """Train the model using cross-validation.

        This method trains the model using cross-validation, according to the
        specified cross-validation splitter. The trained models can be accessed
        through the `raw_model` property.

        Parameters
        ----------
        train_dataset : T
            The training dataset.
        n_jobs : int or None, optional
            The number of jobs to run in parallel. If -1, all CPUs are used.
            If None, no parallel processing is used. Default is None.

        Returns
        -------
        None
        """
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
        """Make predictions using the trained models.

        This method makes predictions using the trained models. It supports two
        prediction modes:

        - Out-of-fold (OOF) predictions: Predictions for the training data using
        models trained on other folds.

        - Averaging Ensemble predictions: Predictions using averaging ensemble of
        all trained models.This mode should ONLY be used for unseen data (test
        data), as using it on training data would lead to data leakage.

        Parameters
        ----------
        pred_dataset : T
            The dataset to make predictions for.
        n_jobs : int or None, optional
            The number of jobs to run in parallel. If -1, all CPUs are used.
            If None, no parallel processing is used. Default is None.
        mode : PredMode, optional
            The prediction mode. Can be either PredMode.OOF_PRED for out-of-fold
            predictions or PredMode.AVG_ENSEMBLE for averaging ensemble predictions.
            Default is PredMode.OOF_PRED.

        Returns
        -------
        Prediction
            The predictions as a NumPy array.

        Raises
        ------
        ValueError
            If the prediction mode is invalid.
        """
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
        """Get the raw models from cross-validation.

        Returns
        -------
        RawModels[U]
            The raw models as a RawModels object.
        """
        return self._model

    @overload
    def evaluate[X](
        self,
        y_true: Target,
        y_pred: Prediction,
        eval_func: EvalFunc[X],
        eval_mode: Literal[EvalMode.POOLING] = EvalMode.POOLING,
    ) -> X: ...

    @overload
    def evaluate[X](
        self,
        y_true: Target,
        y_pred: Prediction,
        eval_func: EvalFunc[X],
        eval_mode: Literal[EvalMode.FOLD_WISE],
    ) -> Sequence[X]: ...

    def evaluate[X](
        self,
        y_true: Target,
        y_pred: Prediction,
        eval_func: EvalFunc[X],
        eval_mode: EvalMode = EvalMode.POOLING,
    ) -> X | Sequence[X]:
        """Evaluate the model's predictions against true values.

        This method evaluates predictions from cross-validation models. The predictions
        can be either out-of-fold (OOF) predictions or predictions on unseen data
        (held-out test set).

        Parameters
        ----------
        y_true : Target
            The true target values as a NumPy array.
        y_pred : Prediction
            The predicted values as a NumPy array. Must have the same shape as y_true.
            These can be:
            - Out-of-fold predictions from predict(mode=PredMode.OOF_PRED)
            - Predictions on unseen data from predict(mode=PredMode.AVG_ENSEMBLE)
        eval_func : EvalFunc[X]
            The evaluation function that takes (y_true, y_pred) and returns a metric.
            Common examples include sklearn.metrics functions like r2_score, mae, etc.
        eval_mode : EvalMode, default=EvalMode.POOLING
            The evaluation mode:
            - EvalMode.POOLING: Compute a single metric across all predictions
              (standard for both OOF evaluation and held-out test set evaluation)
            - EvalMode.FOLD_WISE: Compute metrics for each fold separately
              (useful for analyzing per-fold performance in OOF predictions)

        Returns
        -------
        X | Sequence[X]
            If eval_mode is POOLING, returns a single evaluation score of type X.
            If eval_mode is FOLD_WISE, returns a list of evaluation scores per fold.

        Raises
        ------
        ValueError
            If y_true or y_pred are not NumPy arrays.
        """
        if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
            raise ValueError(
                f"Both y_true and y_pred must be numpy arrays, got {type(y_true)} and {type(y_pred)}"
            )

        match eval_mode:
            case EvalMode.POOLING:
                return eval_func(y_true, y_pred)
            case EvalMode.FOLD_WISE:
                return [
                    eval_func(y_true[index], y_pred[index])
                    for index in self.cv_indices.test
                ]
            case _:
                raise ValueError(f"Invalid evaluation mode: {eval_mode}")

    @property
    def train_config(self) -> V:
        """Get the training configuration.

        Returns
        -------
        V
            The training configuration.
        """
        return self._train_config

    @train_config.setter
    def train_config(self, config: V) -> None:
        """Set the training configuration.

        Parameters
        ----------
        config : V
            The new training configuration.
        """
        self._train_config = config

    @property
    def pred_config(self) -> W:
        """Get the prediction configuration.

        Returns
        -------
        W
            The prediction configuration.
        """
        return self._pred_config

    @pred_config.setter
    def pred_config(self, config: W) -> None:
        """Set the prediction configuration.

        Parameters
        ----------
        config : W
            The new prediction configuration.
        """
        self._pred_config = config

    @property
    def cv_indices(self) -> SplittedDatasetsIndices:
        """Get the cross-validation split indices after training.

        This property returns the cross-validation split indices that are stored
        in the instance after the `train` method is executed.

        Returns
        -------
        SplittedDatasetsIndices
            The cross-validation split indices.
        """
        return self._cv_indices

    @property
    def k_fold(self) -> _BaseKFold | SplittedDatasetsIndices:
        """Get the cross-validation splitter.

        Returns
        -------
        _BaseKFold or SplittedDatasetsIndices
            The cross-validation splitter.
        """
        return self._k_fold
