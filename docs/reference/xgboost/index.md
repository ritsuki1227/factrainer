# factrainer.xgboost

The `factrainer.xgboost` module will provide integration with the XGBoost framework. It will implement the interfaces defined in `factrainer.base` to allow using XGBoost models with Factrainer's cross-validation functionality.

!!! note "Implementation Status"
    The XGBoost integration is currently a placeholder and not yet implemented.
    Functionality will be added in a future release.

## Future Public API

In the future, this module is expected to provide the following classes:

- XgbDataset: Wrapper for XGBoost datasets
- XgbModelConfig: Factory for XGBoost model configurations
- XgbTrainConfig: Configuration for XGBoost training
- XgbPredictConfig: Configuration for XGBoost prediction
- XgbModel: Wrapper for XGBoost models

## Future Usage Example

```python
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from factrainer.core import CvModelContainer
from factrainer.xgboost import XgbDataset, XgbModelConfig, XgbTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Create dataset
dataset = XgbDataset(X=X, y=y)

# Configure model
config = XgbModelConfig.create(
    train_config=XgbTrainConfig(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        num_boost_round=100,
        early_stopping_rounds=10
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)
