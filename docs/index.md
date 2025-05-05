# Welcome to Factrainer

**Factrainer** (Framework Agnostic Cross-validation Trainer) is a machine learning tool that provides a flexible cross-validation training framework. It addresses the limitations of existing cross-validation utilities in popular ML libraries by offering a unified, parallelized approach that retains models and yields out-of-fold (OOF) predictions.

## Why Use Factrainer?

Various ML frameworks (e.g., Scikit-learn, LightGBM) offer cross-validation functions. However, each has different features and interfaces. The table below highlights some widely used cross-validation APIs and which capabilities they support:

| Framework    | API                 | OOF prediction | return trained models | parallel training |
| ------------ | ------------------- | :------------: | :-------------------: | :---------------: |
| LightGBM     | `lgb.cv`            |       🚫       |          ✅️          |        🚫         |
| Scikit-learn | `GridSearchCV`      |       🚫       |          🚫           |        ✅️        |
| Scikit-learn | `cross_val_score`   |       🚫       |          🚫           |        ✅️        |
| Scikit-learn | `cross_val_predict` |      ✅️       |          🚫           |        ✅️        |
| Scikit-learn | `cross_validate`    |       🚫       |          ✅️          |        ✅️        |

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

At present, LightGBM and Scikit-learn are the primary supported backends. Support for additional frameworks will be added as the project evolves.

## Get started

Code example: **California Housing dataset**

```python
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

data = fetch_california_housing()
dataset = LgbDataset(
    dataset=lgb.Dataset(
        data.data, label=data.target
    )
)
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(
        params={"objective": "regression"},
        callbacks=[lgb.early_stopping(100, verbose=False)],
    ),
)
k_fold = KFold(n_splits=4, shuffle=True, random_state=1)
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# trained models
model.raw_model

# OOF prediction
y_pred = model.predict(dataset, n_jobs=4)
print(r2_score(data.target, y_pred))
```

## Project Status

Factrainer is in active development. The goal is to expand support to more frameworks and make the tool even more robust. Contributions, issues, and feedback are welcome to help shape the future of Factrainer.
