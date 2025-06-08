from pathlib import Path
from typing import Any, Callable, Self

import scipy
from factrainer.base.config import (
    BaseLearner,
    BasePredictConfig,
    BasePredictor,
    BaseTrainConfig,
    MlModelConfig,
)
from factrainer.base.dataset import Prediction
from pydantic import ConfigDict

import lightgbm as lgb
from lightgbm.engine import _LGBM_CustomMetricFunction

from .dataset.dataset import LgbDataset
from .raw_model import LgbModel


class LgbTrainConfig(BaseTrainConfig):
    """Configuration for LightGBM training parameters.

    This class encapsulates all parameters that can be passed to `lgb.train()`,
    except for validation data-related parameters. Validation data handling is
    managed separately: `SingleModelContainer` decides whether to use validation
    data in its train method, while `CvModelContainer` automatically handles it
    during cross-validation.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters for training.
    num_boost_round : int, default=100
        Number of boosting iterations.
    valid_names : list[str] | None, default=None
        Names of ``valid_sets``.
    feval : callable or list of callable, optional
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.
    init_model : str, pathlib.Path, Booster or None, optional
        Filename of LightGBM model or Booster instance used for continue training.
    keep_training_booster : bool, default=False
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
    callbacks : list of callable, optional
        List of callback functions that are applied at each iteration.

    See Also
    --------
    lightgbm.train : The underlying LightGBM training function.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from factrainer.lightgbm import LgbTrainConfig
    >>> config = LgbTrainConfig(
    ...     params={
    ...         "objective": "regression",
    ...         "metric": "rmse",
    ...         "boosting_type": "gbdt",
    ...         "num_leaves": 31,
    ...         "learning_rate": 0.05,
    ...     },
    ...     num_boost_round=100,
    ...     callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)],
    ... )
    """

    params: dict[str, Any]
    num_boost_round: int = 100
    valid_names: list[str] | None = None
    feval: _LGBM_CustomMetricFunction | list[_LGBM_CustomMetricFunction] | None = None
    init_model: str | Path | lgb.Booster | None = None
    keep_training_booster: bool = False
    callbacks: list[Callable[..., Any]] | None = None


class LgbPredictConfig(BasePredictConfig):
    """Configuration for LightGBM prediction parameters.

    This class encapsulates all parameters that can be passed to the `predict()`
    method of a LightGBM Booster, except for the data parameter itself.

    Parameters
    ----------
    start_iteration : int, default=0
        Start index of the iteration to predict.
        If <= 0, starts from the first iteration.
    num_iteration : int | None, default=None
        Total number of iterations used in the prediction.
        - If None: if the best iteration exists and start_iteration <= 0,
          the best iteration is used; otherwise, all iterations from
          start_iteration are used.
        - If <= 0: all iterations from start_iteration are used (no limits).
    raw_score : bool, default=False
        Whether to predict raw scores.
        - If False: returns transformed scores (e.g., probabilities for
          binary classification).
        - If True: returns raw scores before transformation (e.g., raw
          log-odds for binary classification).
    pred_leaf : bool, default=False
        Whether to predict leaf indices.
        - If True: returns the index of the leaf that each sample ends up
          in for each tree. Output shape is [n_samples, n_trees] or
          [n_samples, n_trees * n_classes] for multiclass.
        - If False: returns predicted values.
    pred_contrib : bool, default=False
        Whether to predict feature contributions (SHAP values).
        - If True: returns feature contributions for each prediction,
          including the base value (intercept) as the last column.
          Output shape is [n_samples, n_features + 1] or
          [n_samples, (n_features + 1) * n_classes] for multiclass.
        - If False: returns predicted values.
    data_has_header : bool, default=False
        Whether the data file has a header when data is provided as a
        file path. Only used when prediction data is a string path to
        a text file (CSV, TSV, or LibSVM).
    validate_features : bool, default=False
        Whether to validate that features in the prediction data match
        those used during training. Only applies when prediction data
        is a pandas DataFrame.

    See Also
    --------
    lightgbm.Booster.predict : The underlying LightGBM prediction method.

    Notes
    -----
    - Only one of `pred_leaf` and `pred_contrib` can be True at a time.
    - When using custom objective functions, raw_score=False still returns
      raw predictions since the transformation function is not known.

    Examples
    --------
    >>> from factrainer.lightgbm import LgbPredictConfig
    >>> # Standard prediction
    >>> config = LgbPredictConfig()
    >>> # Raw score prediction
    >>> config = LgbPredictConfig(raw_score=True)
    >>> # Get SHAP values
    >>> config = LgbPredictConfig(pred_contrib=True)
    >>> # Predict leaf indices
    >>> config = LgbPredictConfig(pred_leaf=True)
    """

    start_iteration: int = 0
    num_iteration: int | None = None
    raw_score: bool = False
    pred_leaf: bool = False
    pred_contrib: bool = False
    data_has_header: bool = False
    validate_features: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class LgbLearner(BaseLearner[LgbDataset, LgbModel, LgbTrainConfig]):
    def train(
        self,
        train_dataset: LgbDataset,
        val_dataset: LgbDataset | None,
        config: LgbTrainConfig,
    ) -> LgbModel:
        return LgbModel(
            model=lgb.train(
                **dict(config),
                train_set=train_dataset.dataset,
                valid_sets=[val_dataset.dataset] if val_dataset else None,
            )
        )


class LgbPredictor(BasePredictor[LgbDataset, LgbModel, LgbPredictConfig]):
    def predict(
        self,
        dataset: LgbDataset,
        raw_model: LgbModel,
        config: LgbPredictConfig,
    ) -> Prediction:
        y_pred = raw_model.model.predict(data=dataset.dataset.data, **dict(config))
        if isinstance(y_pred, list):
            raise NotImplementedError
        elif isinstance(y_pred, scipy.sparse.spmatrix):
            raise NotImplementedError
        return y_pred


class LgbModelConfig(
    MlModelConfig[LgbDataset, LgbModel, LgbTrainConfig, LgbPredictConfig],
):
    """Configuration container for LightGBM models in the factrainer framework.

    This class encapsulates all necessary components for training and prediction
    with LightGBM models.

    Warnings
    --------
    Do not instantiate this class directly by calling ``LgbModelConfig(...)``.
    Use the factory method ``LgbModelConfig.create`` instead.

    Attributes
    ----------
    learner : LgbLearner
        Component responsible for training LightGBM models.
    predictor : LgbPredictor
        Component responsible for making predictions with trained models.
    train_config : LgbTrainConfig
        Configuration for training parameters (params, num_boost_round, etc.).
    pred_config : LgbPredictConfig
        Configuration for prediction parameters (iteration range, output type, etc.).

    See Also
    --------
    LgbModelConfig.create : Factory method for creating configurations.
    LgbTrainConfig : Configuration for training parameters.
    LgbPredictConfig : Configuration for prediction parameters.
    SingleModelContainer : For training a single model.
    CvModelContainer : For cross-validation workflows.

    Examples
    --------
    >>> from factrainer.lightgbm import LgbModelConfig, LgbTrainConfig
    >>> # Create configuration with default prediction settings
    >>> train_config = LgbTrainConfig(
    ...     params={"objective": "regression", "metric": "rmse"}, num_boost_round=100
    ... )
    >>> model_config = LgbModelConfig.create(train_config)
    >>> # Create configuration with custom prediction settings
    >>> from factrainer.lightgbm import LgbPredictConfig
    >>> pred_config = LgbPredictConfig(raw_score=True)
    >>> model_config = LgbModelConfig.create(train_config, pred_config)
    """

    learner: LgbLearner
    predictor: LgbPredictor
    train_config: LgbTrainConfig
    pred_config: LgbPredictConfig

    @classmethod
    def create(
        cls, train_config: LgbTrainConfig, pred_config: LgbPredictConfig | None = None
    ) -> Self:
        """Create a new LgbModelConfig instance.

        Parameters
        ----------
        train_config : LgbTrainConfig
            Configuration for training parameters including LightGBM params,
            number of boosting rounds, callbacks, etc.
        pred_config : LgbPredictConfig | None, optional
            Configuration for prediction parameters. If None, uses default
            LgbPredictConfig() which performs standard prediction.

        Returns
        -------
        LgbModelConfig
            A configuration instance ready for use with
            SingleModelContainer or CvModelContainer.
        """
        return cls(
            learner=LgbLearner(),
            predictor=LgbPredictor(),
            train_config=train_config,
            pred_config=pred_config if pred_config is not None else LgbPredictConfig(),
        )
