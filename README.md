# factrainer

![CI](https://github.com/ritsuki1227/factrainer/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/ritsuki1227/factrainer/branch/main/graph/badge.svg)](https://codecov.io/gh/ritsuki1227/factrainer)
[![PyPI](https://img.shields.io/pypi/v/factrainer.svg)](https://pypi.python.org/project/factrainer)
[![image](https://img.shields.io/pypi/pyversions/factrainer.svg)](https://pypi.python.org/pypi/factrainer)
![License](https://img.shields.io/github/license/ritsuki1227/factrainer.svg)
![Stars](https://img.shields.io/github/stars/ritsuki1227/factrainer.svg?style=social)

**factrainer** (Framework Agnostic Cross-validation Trainer) is a machine learning tool that provides a flexible cross-validation training framework. It addresses the limitations of existing cross-validation utilities in popular ML libraries by offering a unified, parallelized approach that retains models and yields out-of-fold predictions.

## Why Use factrainer?

Modern ML frameworks have useful cross-validation functions, but they come with notable limitations:

- `scikit-learn`:
  - `cross_val_score`: cannot provide out-of-fold (OOF) predictions for each sample.
  - `cross_val_predict`: cannot retain the trained model from each fold (only returns predictions).
- `LightGBM`:
  - `lgb.cv`: does not support parallelized training of cv.

These gaps make it cumbersome to get both OOF predictions and reusable trained models in a single workflow.

## Key Features

- **Unified Cross-Validation API** – Provides a single, consistent interface to perform K-fold (or any CV) training, acting as a meta-framework that wraps around multiple ML libraries.
- **Parallelized Training** – Run cross-validation folds in parallel to fully utilize multi-core CPUs and speed up model training.
- **Mutable Model Container** – Access each fold’s trained model as a mutable object. This makes it easy to analyze models or create ensembles from the fold models.
- **Out-of-Fold Predictions** – Retrieve out-of-fold predictions for every training instance through a simple API.

## Installation

To install with LightGBM support:

```sh
pip install "factrainer[lightgbm]"
```

At present, LightGBM is the primary supported backend. Support for additional frameworks will be added as the project evolves.

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

factrainer is in active development. The goal is to expand support to more frameworks and make the tool even more robust. Contributions, issues, and feedback are welcome to help shape the future of factrainer.
