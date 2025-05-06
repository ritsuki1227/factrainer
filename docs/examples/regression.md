# Regression Examples

This page provides examples of using Factrainer for regression tasks.

## Simple Regression

### California Housing Regression with LightGBM

This example demonstrates how to use Factrainer with LightGBM for regression on the California Housing dataset.

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
train_dataset = LgbDataset(
    dataset=lgb.Dataset(X_train, label=y_train)
)
test_dataset = LgbDataset(
    dataset=lgb.Dataset(X_test, label=y_test)
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
model.train(train_dataset, n_jobs=4)

# Get OOF predictions
y_pred_oof = model.predict(train_dataset, n_jobs=4)
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_oof)):.4f}")
print(f"OOF R²: {r2_score(y_train, y_pred_oof):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
```

### Diabetes Regression with scikit-learn

This example demonstrates how to use Factrainer with scikit-learn for regression on the Diabetes dataset.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from factrainer.core import CvModelContainer
from factrainer.sklearn import SklearnDataset, SklearnModelConfig, SklearnTrainConfig

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets
train_dataset = SklearnDataset(X=X_train, y=y_train)
test_dataset = SklearnDataset(X=X_test, y=y_test)

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
model.train(train_dataset, n_jobs=4)

# Get OOF predictions
y_pred_oof = model.predict(train_dataset, n_jobs=4)
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_oof)):.4f}")
print(f"OOF R²: {r2_score(y_train, y_pred_oof):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
```

## Multiple Regression

### Boston Housing Regression with LightGBM

This example demonstrates how to use Factrainer with LightGBM for regression on the Boston Housing dataset.

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
train_dataset = LgbDataset(
    dataset=lgb.Dataset(X_train, label=y_train)
)
test_dataset = LgbDataset(
    dataset=lgb.Dataset(X_test, label=y_test)
)

# Configure model
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        },
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20, verbose=False)],
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(train_dataset, n_jobs=4)

# Get OOF predictions
y_pred_oof = model.predict(train_dataset, n_jobs=4)
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_oof)):.4f}")
print(f"OOF R²: {r2_score(y_train, y_pred_oof):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.raw_model.models[0].model.feature_importance()
}).sort_values('Importance', ascending=False)
print(feature_importance)
```

## scikit-learn Examples

### Feature Engineering

This example demonstrates how to use Factrainer with scikit-learn for feature engineering in regression tasks.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from factrainer.core import CvModelContainer
from factrainer.sklearn import SklearnDataset, SklearnModelConfig, SklearnTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Convert to DataFrame for better feature handling
X_df = pd.DataFrame(X, columns=data.feature_names)

# Add some categorical features for demonstration
X_df['MedInc_Cat'] = pd.qcut(X_df['MedInc'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
X_df['AveOccup_Cat'] = pd.qcut(X_df['AveOccup'], 3, labels=['Low', 'Medium', 'High'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Define feature types
numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
categorical_features = ['MedInc_Cat', 'AveOccup_Cat']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Create datasets
train_dataset = SklearnDataset(X=X_train, y=y_train)
test_dataset = SklearnDataset(X=X_test, y=y_test)

# Configure model
config = SklearnModelConfig.create(
    train_config=SklearnTrainConfig(
        estimator=pipeline
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(train_dataset, n_jobs=4)

# Get OOF predictions
y_pred_oof = model.predict(train_dataset, n_jobs=4)
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_oof)):.4f}")
print(f"OOF R²: {r2_score(y_train, y_pred_oof):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
```

### Stacking Regression

This example demonstrates how to use Factrainer with scikit-learn for stacking regression.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from factrainer.core import CvModelContainer
from factrainer.sklearn import SklearnDataset, SklearnModelConfig, SklearnTrainConfig

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets
train_dataset = SklearnDataset(X=X_train, y=y_train)
test_dataset = SklearnDataset(X=X_test, y=y_test)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train base models and get OOF predictions
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0, random_state=42)),
    ('elasticnet', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
]

oof_predictions = np.zeros((X_train.shape[0], len(base_models)))
test_predictions = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, estimator) in enumerate(base_models):
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])
    
    # Configure model
    config = SklearnModelConfig.create(
        train_config=SklearnTrainConfig(
            estimator=pipeline
        ),
    )
    
    # Create and train model
    model = CvModelContainer(config, k_fold)
    model.train(train_dataset, n_jobs=4)
    
    # Get OOF predictions
    oof_predictions[:, i] = model.predict(train_dataset, n_jobs=4)
    
    # Get test predictions
    test_predictions[:, i] = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
    
    # Print model performance
    print(f"{name} - OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof_predictions[:, i])):.4f}")
    print(f"{name} - OOF R²: {r2_score(y_train, oof_predictions[:, i]):.4f}")

# Create meta-model dataset
meta_train_dataset = SklearnDataset(X=oof_predictions, y=y_train)
meta_test_dataset = SklearnDataset(X=test_predictions, y=y_test)

# Configure meta-model
meta_config = SklearnModelConfig.create(
    train_config=SklearnTrainConfig(
        estimator=Ridge(alpha=1.0, random_state=42)
    ),
)

# Create and train meta-model
meta_model = CvModelContainer(meta_config, k_fold)
meta_model.train(meta_train_dataset, n_jobs=4)

# Get meta-model predictions
meta_pred_oof = meta_model.predict(meta_train_dataset, n_jobs=4)
meta_pred_test = meta_model.predict(meta_test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")

# Print meta-model performance
print(f"Meta-model - OOF RMSE: {np.sqrt(mean_squared_error(y_train, meta_pred_oof)):.4f}")
print(f"Meta-model - OOF R²: {r2_score(y_train, meta_pred_oof):.4f}")
print(f"Meta-model - Test RMSE: {np.sqrt(mean_squared_error(y_test, meta_pred_test)):.4f}")
print(f"Meta-model - Test R²: {r2_score(y_test, meta_pred_test):.4f}")
