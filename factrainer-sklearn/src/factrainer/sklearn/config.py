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

from .dataset.dataset import SklearnDataset
from .raw_model import Predictable, ProbPredictable, SklearnModel


class SklearnTrainConfig(BaseTrainConfig):
    """Configuration for scikit-learn training parameters.

    This class encapsulates the estimator to be trained. Additional keyword
    arguments for the estimator's `fit()` method can be passed as attributes
    of this configuration.

    Parameters
    ----------
    estimator : Predictable | ProbPredictable
        A scikit-learn estimator instance that implements the fit method.
        Must also implement either predict or predict_proba method.

    See Also
    --------
    sklearn.base.BaseEstimator : The base class for all scikit-learn estimators.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from factrainer.sklearn import SklearnTrainConfig
    >>> # Basic configuration
    >>> estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    >>> config = SklearnTrainConfig(estimator=estimator)
    >>> # With additional fit keyword arguments
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDClassifier
    >>> sample_weights = np.array([1, 2, 1, 1, 2])
    >>> estimator = SGDClassifier()
    >>> config = SklearnTrainConfig(
    ...     estimator=estimator,
    ...     sample_weight=sample_weights,  # passed as kwargs to fit()
    ... )
    """

    estimator: Predictable | ProbPredictable
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SklearnPredictMethod(Enum):
    """Prediction method selection for scikit-learn models.

    Attributes
    ----------
    AUTO : auto
        Automatically selects predict_proba if available, otherwise predict.
    PREDICT : auto
        Uses the predict method (returns class labels or regression values).
    PREDICT_PROBA : auto
        Uses the predict_proba method (returns probability estimates).
    """

    AUTO = auto()
    PREDICT = auto()
    PREDICT_PROBA = auto()


class SklearnPredictConfig(BasePredictConfig):
    """Configuration for scikit-learn prediction parameters.

    This class encapsulates parameters for prediction with scikit-learn models.
    Additional keyword arguments for the estimator's prediction methods can be
    passed as attributes of this configuration.

    Parameters
    ----------
    predict_method : SklearnPredictMethod, default=SklearnPredictMethod.AUTO
        The prediction method to use:

        - AUTO: Automatically selects predict_proba if available, otherwise predict.
        - PREDICT: Uses the predict method (returns class labels or regression values).
        - PREDICT_PROBA: Uses the predict_proba method (returns probability estimates).

    See Also
    --------
    SklearnPredictMethod : Enum of available prediction methods.

    Examples
    --------
    >>> from factrainer.sklearn import SklearnPredictConfig, SklearnPredictMethod
    >>> # Default configuration (auto-selects method)
    >>> config = SklearnPredictConfig()
    >>> # Force using predict method
    >>> config = SklearnPredictConfig(predict_method=SklearnPredictMethod.PREDICT)
    >>> # Use predict_proba for probability estimates
    >>> config = SklearnPredictConfig(predict_method=SklearnPredictMethod.PREDICT_PROBA)
    """

    predict_method: SklearnPredictMethod = SklearnPredictMethod.AUTO
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
            case SklearnPredictMethod.PREDICT_PROBA:
                if not hasattr(raw_model.estimator, "predict_proba"):
                    raise ValueError("The model does not support predict_proba method.")
                return raw_model.estimator.predict_proba(
                    dataset.X,
                    **(config.model_extra if config.model_extra is not None else {}),
                )
            case SklearnPredictMethod.PREDICT:
                if not hasattr(raw_model.estimator, "predict"):
                    raise ValueError("The model does not support predict method.")
                return raw_model.estimator.predict(
                    dataset.X,
                    **(config.model_extra if config.model_extra is not None else {}),
                )
            case SklearnPredictMethod.AUTO:
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
    """Configuration container for scikit-learn models in the factrainer framework.

    This class encapsulates all necessary components for training and prediction
    with scikit-learn models.

    Warnings
    --------
    Do not instantiate this class directly by calling ``SklearnModelConfig(...)``.
    Use the factory method ``SklearnModelConfig.create`` instead.

    Attributes
    ----------
    learner : SklearnLearner
        Component responsible for training scikit-learn models.
    predictor : SklearnPredictor
        Component responsible for making predictions with trained models.
    train_config : SklearnTrainConfig
        Configuration for training parameters (estimator and fit kwargs).
    pred_config : SklearnPredictConfig
        Configuration for prediction parameters (prediction method and kwargs).

    See Also
    --------
    SklearnModelConfig.create : Factory method for creating configurations.
    SklearnTrainConfig : Configuration for training parameters.
    SklearnPredictConfig : Configuration for prediction parameters.
    SingleModelContainer : For training a single model.
    CvModelContainer : For cross-validation workflows.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from factrainer.sklearn import SklearnModelConfig, SklearnTrainConfig
    >>> # Create configuration with default prediction settings
    >>> estimator = RandomForestClassifier(n_estimators=10)
    >>> train_config = SklearnTrainConfig(estimator=estimator)
    >>> model_config = SklearnModelConfig.create(train_config)
    >>> # Create configuration with custom prediction settings
    >>> from factrainer.sklearn import SklearnPredictConfig, SklearnPredictMethod
    >>> pred_config = SklearnPredictConfig(
    ...     predict_method=SklearnPredictMethod.PREDICT_PROBA
    ... )
    >>> model_config = SklearnModelConfig.create(train_config, pred_config)
    """

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
        """Create a new SklearnModelConfig instance.

        Parameters
        ----------
        train_config : SklearnTrainConfig
            Configuration for training parameters including the estimator
            and any additional fit keyword arguments.
        pred_config : SklearnPredictConfig | None, optional
            Configuration for prediction parameters. If None, uses default
            SklearnPredictConfig() which automatically selects the prediction method.

        Returns
        -------
        SklearnModelConfig
            A configuration instance ready for use with
            SingleModelContainer or CvModelContainer.
        """
        return cls(
            learner=SklearnLearner(),
            predictor=SklearnPredictor(),
            train_config=train_config,
            pred_config=pred_config
            if pred_config is not None
            else SklearnPredictConfig(),
        )
