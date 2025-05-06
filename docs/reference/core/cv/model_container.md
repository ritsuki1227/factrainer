# CvModelContainer

The `CvModelContainer` class is the main entry point for cross-validation in Factrainer. It provides a unified interface for training models with cross-validation and making predictions with the trained models.

## Class Definition

```python
class CvModelContainer[
    T: IndexableDataset,
    U: RawModel,
    V: BaseTrainConfig,
    W: BasePredictConfig,
](BaseModelContainer[T, RawModels[U], V, W])
```

## Description

The `CvModelContainer` class is a container for cross-validation models. It takes a model configuration and a cross-validation splitter, and provides methods for training models and making predictions.

The class is generic over four type parameters:
- `T`: The dataset type, which must implement the `IndexableDataset` interface
- `U`: The raw model type, which must implement the `RawModel` interface
- `V`: The training configuration type, which must implement the `BaseTrainConfig` interface
- `W`: The prediction configuration type, which must implement the `BasePredictConfig` interface

## Constructor

```python
def __init__(
    self,
    model_config: BaseMlModelConfig[T, U, V, W],
    k_fold: _BaseKFold | SplittedDatasetsIndices,
) -> None
```

### Parameters

- **model_config**: The model configuration, which includes the learner, predictor, training configuration, and prediction configuration.
- **k_fold**: The cross-validation splitter, which can be either a scikit-learn `_BaseKFold` object or a `SplittedDatasetsIndices` object.

## Methods

### train

```python
def train(self, train_dataset: T, n_jobs: int | None = None) -> None
```

Trains the model using cross-validation.

#### Parameters

- **train_dataset**: The training dataset.
- **n_jobs**: The number of jobs to run in parallel. If `None`, all CPUs are used.

#### Returns

None

### predict

```python
def predict(
    self,
    pred_dataset: T,
    n_jobs: int | None = None,
    mode: PredMode = PredMode.OOF_PRED,
) -> Prediction
```

Makes predictions using the trained models.

#### Parameters

- **pred_dataset**: The dataset to make predictions for.
- **n_jobs**: The number of jobs to run in parallel. If `None`, all CPUs are used.
- **mode**: The prediction mode. Can be either `PredMode.OOF_PRED` for out-of-fold predictions or `PredMode.AVG_ENSEMBLE` for ensemble predictions.

#### Returns

The predictions as a NumPy array.

## Properties

### raw_model

```python
@property
def raw_model(self) -> RawModels[U]
```

Gets the raw models from cross-validation.

#### Returns

The raw models as a `RawModels` object.

### train_config

```python
@property
def train_config(self) -> V
```

Gets the training configuration.

#### Returns

The training configuration.

### train_config (setter)

```python
@train_config.setter
def train_config(self, config: V) -> None
```

Sets the training configuration.

#### Parameters

- **config**: The new training configuration.

#### Returns

None

### pred_config

```python
@property
def pred_config(self) -> W
```

Gets the prediction configuration.

#### Returns

The prediction configuration.

### pred_config (setter)

```python
@pred_config.setter
def pred_config(self, config: W) -> None
```

Sets the prediction configuration.

#### Parameters

- **config**: The new prediction configuration.

#### Returns

None

### cv_indices

```python
@property
def cv_indices(self) -> SplittedDatasetsIndices
```

Gets the cross-validation indices.

#### Returns

The cross-validation indices as a `SplittedDatasetsIndices` object.

### k_fold

```python
@property
def k_fold(self) -> _BaseKFold | SplittedDatasetsIndices
```

Gets the cross-validation splitter.

#### Returns

The cross-validation splitter, which can be either a scikit-learn `_BaseKFold` object or a `SplittedDatasetsIndices` object.

## Examples

### Basic Usage

```python
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
data = fetch_california_housing()
dataset = LgbDataset(
    dataset=lgb.Dataset(
        data.data, label=data.target
    )
)

# Configure model
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={"objective": "regression"},
        callbacks=[lgb.early_stopping(100, verbose=False)],
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=4, shuffle=True, random_state=1)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)
```

### Using Different Prediction Modes

```python
# Get OOF predictions
y_pred_oof = model.predict(dataset, n_jobs=4, mode=PredMode.OOF_PRED)

# Get ensemble predictions
y_pred_ensemble = model.predict(dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
```

### Accessing Trained Models

```python
# Get the raw models
raw_models = model.raw_model

# Access individual models
for i, raw_model in enumerate(raw_models.models):
    print(f"Model {i}: {raw_model}")
```

### Changing Configurations

```python
# Change the training configuration
model.train_config = LgbTrainConfig(
    params={"objective": "regression", "learning_rate": 0.1},
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

# Change the prediction configuration
model.pred_config = LgbPredictConfig(
    num_iteration=100
)
