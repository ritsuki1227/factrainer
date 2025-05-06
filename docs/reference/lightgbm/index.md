# factrainer.lightgbm

The `factrainer.lightgbm` module provides integration with the LightGBM framework. It implements the interfaces defined in `factrainer.base` to allow using LightGBM models with Factrainer's cross-validation functionality.

## Public API

The following classes and functions are part of the public API of the `factrainer.lightgbm` module:

- [LgbDataset](lgbdataset.md): Wrapper for LightGBM datasets
- [LgbModelConfig](lgbmodelconfig.md): Factory for LightGBM model configurations
- [LgbTrainConfig](lgbtrainconfig.md): Configuration for LightGBM training
- [LgbPredictConfig](lgbpredictconfig.md): Configuration for LightGBM prediction
- [LgbModel](lgbmodel.md): Wrapper for LightGBM models

## Usage Example

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
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
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
