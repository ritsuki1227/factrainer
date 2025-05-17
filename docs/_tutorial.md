# Tutorial

This tutorial will guide you through the basics of using Factrainer for cross-validation in machine learning tasks.

## Installation

First, install Factrainer with the desired backends:

```bash
# Install with LightGBM and scikit-learn support
pip install "factrainer[lightgbm,sklearn]"

# Or install with all supported backends
pip install "factrainer[all]"
```

## Basic Concepts

Factrainer is designed around a few key concepts:

1. **Datasets**: Wrappers for data that can be used with different ML frameworks
2. **Model Configurations**: Configuration objects that define how models are trained and used for prediction
3. **Model Containers**: Objects that hold trained models and provide methods for training and prediction

## Quick Start

Let's start with a simple example using LightGBM for regression:

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Create dataset
dataset = LgbDataset(
    dataset=lgb.Dataset(X, label=y)
)

# Configure model
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1
        },
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10, verbose=False)],
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)
print(f"R² score: {r2_score(y, y_pred):.4f}")
```

## Working with Different Frameworks

Factrainer provides a unified API for working with different ML frameworks. Here's how to use it with scikit-learn:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from factrainer.core import CvModelContainer
from factrainer.sklearn import SklearnDataset, SklearnModelConfig, SklearnTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Create dataset
dataset = SklearnDataset(X=X, y=y)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Configure model
config = SklearnModelConfig.create(
    train_config=SklearnTrainConfig(
        estimator=pipeline,
        fit_params={
            'model__n_jobs': -1
        }
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)
print(f"R² score: {r2_score(y, y_pred):.4f}")
```

## Advanced Features

### Out-of-Fold Predictions vs. Ensemble Predictions

Factrainer supports two prediction modes:

1. **Out-of-Fold (OOF) Predictions**: Predictions for the training data using models trained on other folds
2. **Ensemble Predictions**: Predictions using an ensemble of all trained models

```python
# Get OOF predictions (default)
y_pred_oof = model.predict(dataset, n_jobs=4)

# Get ensemble predictions
from factrainer.core.cv.config import PredMode
y_pred_ensemble = model.predict(dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
```

### Accessing Trained Models

You can access the trained models directly:

```python
# Get the raw models
raw_models = model.raw_model

# Access individual models
for i, raw_model in enumerate(raw_models.models):
    print(f"Model {i}: {raw_model}")
```

### Customizing Training and Prediction

You can customize the training and prediction process by modifying the configuration:

```python
# Change the training configuration
model.train_config = LgbTrainConfig(
    params={
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,  # Changed from 0.1
        "num_leaves": 63,       # Changed from 31
        "verbose": -1
    },
    num_boost_round=200,        # Changed from 100
    callbacks=[lgb.early_stopping(20, verbose=False)],  # Changed from 10
)

# Change the prediction configuration
from factrainer.lightgbm import LgbPredictConfig
model.pred_config = LgbPredictConfig(
    num_iteration=100
)
```

## Working with Different Data Types

Factrainer supports various data types through its dataset wrappers:

### LightGBM

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from factrainer.lightgbm import LgbDataset

# NumPy array
X_np = np.array(...)
y_np = np.array(...)
dataset_np = LgbDataset(
    dataset=lgb.Dataset(X_np, label=y_np)
)

# Pandas DataFrame
X_pd = pd.DataFrame(...)
y_pd = pd.Series(...)
dataset_pd = LgbDataset(
    dataset=lgb.Dataset(X_pd, label=y_pd)
)

# With additional parameters
dataset = LgbDataset(
    dataset=lgb.Dataset(
        X,
        label=y,
        weight=sample_weights,
        group=group_info,
        init_score=init_scores,
        feature_name=feature_names,
        categorical_feature=categorical_features,
    )
)
```

### scikit-learn

```python
import numpy as np
import pandas as pd
from factrainer.sklearn import SklearnDataset

# NumPy array
X_np = np.array(...)
y_np = np.array(...)
dataset_np = SklearnDataset(X=X_np, y=y_np)

# Pandas DataFrame
X_pd = pd.DataFrame(...)
y_pd = pd.Series(...)
dataset_pd = SklearnDataset(X=X_pd, y=y_pd)

# With sample weights
dataset = SklearnDataset(X=X, y=y, sample_weight=sample_weights)
```

## Next Steps

Now that you've learned the basics of Factrainer, you can:

- Check out the [API Reference](reference/index.md) for detailed documentation of all classes and methods
- Explore the [Examples](examples/index.md) for more advanced usage patterns
- Learn about the [Plugin Architecture](reference/base/index.md) if you want to extend Factrainer with your own ML framework
