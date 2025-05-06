# Classification Examples

This page provides examples of using Factrainer for classification tasks.

## Binary Classification

### Breast Cancer Classification with LightGBM

This example demonstrates how to use Factrainer with LightGBM for binary classification on the Breast Cancer Wisconsin dataset.

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
X, y = load_breast_cancer(return_X_y=True)
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
            "objective": "binary",
            "metric": "binary_logloss",
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
print(f"OOF ROC AUC: {roc_auc_score(y_train, y_pred_oof):.4f}")
print(f"OOF Accuracy: {accuracy_score(y_train, y_pred_oof > 0.5):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred_test):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test > 0.5):.4f}")
```

### Ionosphere Classification with scikit-learn

This example demonstrates how to use Factrainer with scikit-learn for binary classification on the Ionosphere dataset.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from factrainer.core import CvModelContainer
from factrainer.sklearn import SklearnDataset, SklearnModelConfig, SklearnTrainConfig

# Load data
X, y = fetch_openml("ionosphere", return_X_y=True, as_frame=False)
y = (y == "g").astype(int)  # Convert to binary (0/1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets
train_dataset = SklearnDataset(X=X_train, y=y_train)
test_dataset = SklearnDataset(X=X_test, y=y_test)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
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
print(f"OOF ROC AUC: {roc_auc_score(y_train, y_pred_oof):.4f}")
print(f"OOF Accuracy: {accuracy_score(y_train, y_pred_oof > 0.5):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred_test):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test > 0.5):.4f}")
```

## Multi-class Classification

### Iris Classification with LightGBM

This example demonstrates how to use Factrainer with LightGBM for multi-class classification on the Iris dataset.

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
X, y = load_iris(return_X_y=True)
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
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
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
y_pred_oof_class = np.argmax(y_pred_oof, axis=1)
print(f"OOF Accuracy: {accuracy_score(y_train, y_pred_oof_class):.4f}")

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
y_pred_test_class = np.argmax(y_pred_test, axis=1)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test_class):.4f}")
```

## LightGBM Examples

### Custom Metrics

This example demonstrates how to use custom metrics with LightGBM in Factrainer.

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
train_dataset = LgbDataset(
    dataset=lgb.Dataset(X_train, label=y_train)
)
test_dataset = LgbDataset(
    dataset=lgb.Dataset(X_test, label=y_test)
)

# Define custom metric
def custom_accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = preds > 0.5
    return 'custom_accuracy', np.mean(labels == preds), True

# Configure model
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1
        },
        num_boost_round=100,
        feval=custom_accuracy,
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

# Get test predictions
y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
```

## Hyperparameter Tuning

This example demonstrates how to use Factrainer with hyperparameter tuning.

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
train_dataset = LgbDataset(
    dataset=lgb.Dataset(X_train, label=y_train)
)
test_dataset = LgbDataset(
    dataset=lgb.Dataset(X_test, label=y_test)
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid
param_grid = [
    {"learning_rate": 0.01, "num_leaves": 31},
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.1, "num_leaves": 31},
    {"learning_rate": 0.01, "num_leaves": 63},
    {"learning_rate": 0.05, "num_leaves": 63},
    {"learning_rate": 0.1, "num_leaves": 63},
]

# Train models with different hyperparameters
results = []
for params in param_grid:
    # Configure model
    config = LgbModelConfig.create(
        train_config=LgbTrainConfig(
            params={
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1
            },
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10, verbose=False)],
        ),
    )
    
    # Create and train model
    model = CvModelContainer(config, k_fold)
    model.train(train_dataset, n_jobs=4)
    
    # Get OOF predictions
    y_pred_oof = model.predict(train_dataset, n_jobs=4)
    oof_auc = roc_auc_score(y_train, y_pred_oof)
    
    # Get test predictions
    y_pred_test = model.predict(test_dataset, n_jobs=4, mode="AVG_ENSEMBLE")
    test_auc = roc_auc_score(y_test, y_pred_test)
    
    results.append({
        "params": params,
        "oof_auc": oof_auc,
        "test_auc": test_auc
    })

# Find best model
best_model = max(results, key=lambda x: x["oof_auc"])
print(f"Best model: {best_model['params']}")
print(f"OOF AUC: {best_model['oof_auc']:.4f}")
print(f"Test AUC: {best_model['test_auc']:.4f}")
