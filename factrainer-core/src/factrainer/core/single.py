import numpy as np
from factrainer.base.config import (
    BaseMlModelConfig,
    BasePredictConfig,
    BaseTrainConfig,
)
from factrainer.base.dataset import BaseDataset, Prediction, Target
from factrainer.base.raw_model import RawModel

from .model_container import BaseModelContainer, EvalFunc


class SingleModelContainer[
    T: BaseDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModelContainer[T, U, V, W]):
    """Single model container for machine learning models.

    This class provides a container for a single machine learning model. It takes a model
    configuration and provides methods for training a model and making predictions.

    Parameters
    ----------
    model_config : BaseMlModelConfig
        The model configuration, which includes the learner, predictor, training
        configuration, and prediction configuration.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.metrics import r2_score
    >>> from sklearn.model_selection import train_test_split
    >>> from factrainer.core import SingleModelContainer
    >>> from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig
    >>>
    >>> # Load data
    >>> X, y = make_regression()
    >>> train_X, test_X, train_y, test_y = train_test_split(
    ...     X, y, test_size=0.2, random_state=1
    ... )
    >>>
    >>> # Create datasets
    >>> train_dataset = LgbDataset(dataset=lgb.Dataset(train_X, train_y))
    >>> val_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    >>> test_dataset = LgbDataset(dataset=lgb.Dataset(test_X, test_y))
    >>>
    >>> # Configure model
    >>> config = LgbModelConfig.create(
    ...     train_config=LgbTrainConfig(
    ...         params={
    ...             "objective": "regression",
    ...             "seed": 1,
    ...             "deterministic": True,
    ...             "verbose": -1,
    ...         },
    ...         callbacks=[lgb.early_stopping(100, verbose=False)],
    ...     ),
    ... )
    >>>
    >>> # Create and train model
    >>> model = SingleModelContainer(config)
    >>> model.train(train_dataset, val_dataset)
    >>>
    >>> # Make predictions
    >>> y_pred = model.predict(test_dataset)
    >>>
    >>> # Evaluate predictions
    >>> metric = model.evaluate(test_y, y_pred, r2_score)
    """

    def __init__(
        self,
        model_config: BaseMlModelConfig[T, U, V, W],
    ) -> None:
        self._learner = model_config.learner
        self._predictor = model_config.predictor
        self._train_config = model_config.train_config
        self._pred_config = model_config.pred_config

    def train(self, train_dataset: T, val_dataset: T | None = None) -> None:
        """Train the model. The trained model can be accessed through
        the `raw_model` property.

        Parameters
        ----------
        train_dataset : T
            The training dataset.
        val_dataset : T | None, optional
            The validation dataset if needed.

        Returns
        -------
        None
        """
        self._model = self._learner.train(train_dataset, val_dataset, self.train_config)

    def predict(self, pred_dataset: T) -> Prediction:
        """Make predictions using the trained model.

        Parameters
        ----------
        pred_dataset : T
            The test dataset.

        Returns
        -------
        Prediction
            The predictions as a NumPy array.
        """
        return self._predictor.predict(pred_dataset, self.raw_model, self.pred_config)

    @property
    def raw_model(self) -> U:
        """Get the trained raw model.

        Returns
        -------
        U
            The trained model as a RawModel object.
        """
        return self._model

    def evaluate[X](
        self,
        y_true: Target,
        y_pred: Prediction,
        eval_func: EvalFunc[X],
    ) -> X:
        """Evaluate the model's predictions against true values.

        This method evaluates predictions from a single trained model, typically
        on a held-out test set or validation set.

        Parameters
        ----------
        y_true : Target
            The true target values as a NumPy array.
        y_pred : Prediction
            The predicted values as a NumPy array. Must have the same shape as y_true.
        eval_func : EvalFunc[X]
            The evaluation function that takes (y_true, y_pred) and returns a metric.
            Common examples include sklearn.metrics functions like r2_score, mae, etc.

        Returns
        -------
        X
            The evaluation score of type X, as returned by eval_func.

        Raises
        ------
        ValueError
            If y_true or y_pred are not NumPy arrays.
        """
        if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
            raise ValueError(
                f"Both y_true and y_pred must be numpy arrays, got {type(y_true)} and {type(y_pred)}"
            )
        return eval_func(y_true, y_pred)

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
