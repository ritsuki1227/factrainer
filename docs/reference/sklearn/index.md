# factrainer.sklearn

The `factrainer.sklearn` module provides integration with the scikit-learn framework. It implements the interfaces defined in `factrainer.base` to allow using scikit-learn models with Factrainer's cross-validation functionality.

## Public API

The following classes and functions are part of the public API of the `factrainer.sklearn` module:

- [SklearnDataset](sklearndataset.md): Wrapper for scikit-learn datasets
- [SklearnModelConfig](sklearnmodelconfig.md): Factory for scikit-learn model configurations
- [SklearnTrainConfig](sklearntrainconfig.md): Configuration for scikit-learn training
- [SklearnPredictConfig](sklearnpredictconfig.md): Configuration for scikit-learn prediction
- [SklearnPredictMethod](sklearnpredictmethod.md): Enumeration of scikit-learn prediction methods
- [SklearnModel](sklearnmodel.md): Wrapper for scikit-learn models

## Usage Example

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
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
