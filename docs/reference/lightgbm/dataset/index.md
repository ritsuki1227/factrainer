# factrainer.lightgbm.dataset

The `factrainer.lightgbm.dataset` module provides classes for handling LightGBM datasets in Factrainer. It implements the `IndexableDataset` interface to allow using LightGBM datasets with Factrainer's cross-validation functionality.

## Files

- [dataset.py](dataset.md): LightGBM dataset wrapper
- [slicer.py](slicer.md): LightGBM dataset slicer
- [types.py](types.md): Type utilities for LightGBM datasets

## Key Classes

### Dataset

- [LgbDataset](dataset.md#lgbdataset): Wrapper for LightGBM datasets that implements the `IndexableDataset` interface

### Slicer

- [LgbDatasetSlicer](slicer.md#lgbdatasetslicer): Slicer for LightGBM datasets
- [DataSlicer](slicer.md#dataslicer): Slicer for the data component of LightGBM datasets
- [LabelSlicer](slicer.md#labelslicer): Slicer for the label component of LightGBM datasets
- [WeightSlicer](slicer.md#weightslicer): Slicer for the weight component of LightGBM datasets
- [GroupSlicer](slicer.md#groupslicer): Slicer for the group component of LightGBM datasets
- [InitScoreSlicer](slicer.md#initscoreslicer): Slicer for the init score component of LightGBM datasets
- [PositionSlicer](slicer.md#positionslicer): Slicer for the position component of LightGBM datasets

### Types

- [IsPdDataFrame](types.md#ispdataframe): Type checker for pandas DataFrames

## Usage Example

```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from factrainer.lightgbm import LgbDataset

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Create LightGBM dataset
lgb_dataset = lgb.Dataset(X, label=y)

# Create Factrainer dataset
dataset = LgbDataset(dataset=lgb_dataset)

# Use indexing to get a subset of the dataset
subset_indices = [0, 1, 2, 3, 4]
subset = dataset[subset_indices]

# Use with cross-validation
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in dataset.k_fold_split(k_fold):
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    
    # Use train_dataset and val_dataset for training
    ...
```

## Advanced Features

### Working with Different Data Types

LightGBM supports various data types for the input data, and `LgbDataset` can handle them:

```python
# NumPy array
X_np = np.array(...)
dataset_np = LgbDataset(dataset=lgb.Dataset(X_np, label=y))

# Pandas DataFrame
import pandas as pd
X_pd = pd.DataFrame(...)
dataset_pd = LgbDataset(dataset=lgb.Dataset(X_pd, label=y))

# SciPy sparse matrix
import scipy.sparse as sp
X_sp = sp.csr_matrix(...)
dataset_sp = LgbDataset(dataset=lgb.Dataset(X_sp, label=y))
```

### Additional Dataset Parameters

You can pass additional parameters to the LightGBM dataset:

```python
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
