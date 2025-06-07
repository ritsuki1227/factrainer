# Factrainer

**Factrainer** (Framework Agnostic Cross-validation Trainer) is a machine learning tool that provides a flexible cross-validation training framework. It addresses the limitations of existing cross-validation utilities in popular ML libraries by offering a unified, parallelized approach that retains models and yields out-of-fold (OOF) predictions.

## Why Use Factrainer?

Various ML frameworks (e.g., Scikit-learn, LightGBM) offer cross-validation functions. However, each has different features and interfaces. The table below highlights some widely used cross-validation APIs and which capabilities they support:

| Framework    | API                                                                                                                     | OOF prediction | return trained models | parallel training |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- | :------------: | :-------------------: | :---------------: |
| LightGBM     | [`lgb.cv`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.cv.html)                                        |       🚫       |          ✅️          |        🚫         |
| Scikit-learn | [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)           |       🚫       |          🚫           |        ✅️        |
| Scikit-learn | [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)     |       🚫       |          🚫           |        ✅️        |
| Scikit-learn | [`cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |      ✅️       |          🚫           |        ✅️        |
| Scikit-learn | [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)       |       🚫       |          ✅️          |        ✅️        |

No built-in API combines OOF predictions, trained-model access, and parallelized training—Factrainer does.

## Key Features

- **Unified Cross-Validation API** – Provides a single, consistent interface to perform K-fold (or any CV) training, acting as a meta-framework that wraps around multiple ML libraries.
- **Parallelized Training** – Run cross-validation folds in parallel to fully utilize multi-core CPUs and speed up model training.
- **Mutable Model Container** – Access each fold’s trained model as a mutable object. This makes it easy to analyze models or create ensembles from the fold models.
- **Out-of-Fold Predictions** – Retrieve out-of-fold predictions for every training instance through a simple API.

## Installation

To install with LightGBM and Scikit-learn support:

```sh
pip install "factrainer[lightgbm,sklearn]"
```

To install with all supported backends (LightGBM, Scikit-learn, XGBoost, and CatBoost):

```sh
pip install "factrainer[all]"
```

At present, LightGBM and Scikit-learn are the primary supported backends. Support for additional frameworks will be implemented as the project evolves.

## Get started

Code example: **California Housing dataset**

```python
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from factrainer.core import CvModelContainer, EvalMode
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

data = fetch_california_housing()
dataset = LgbDataset(
    dataset=lgb.Dataset(
        data.data, label=data.target
    )
)
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={"objective": "regression", "verbose": -1},
        callbacks=[lgb.early_stopping(100, verbose=False)],
    ),
)
k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)

# Evaluate predictions
metric = model.evaluate(data.target, y_pred, r2_score)
print(f"R2 Score: {metric:.4f}")

# Or get per-fold metrics
metrics = model.evaluate(
    data.target, y_pred, r2_score, eval_mode=EvalMode.FOLD_WISE
)
print(f"R2 Scores by fold: {[f'{m:.4f}' for m in metrics]}")

# Access trained models
model.raw_model
```

## Project Status

Factrainer is in active development. The goal is to expand support to more frameworks and make the tool even more robust. Contributions, issues, and feedback are welcome to help shape the future of Factrainer.
