# factrainer.core.cv

The `factrainer.core.cv` module provides the cross-validation functionality in Factrainer. It contains classes for handling cross-validation datasets, models, and prediction.

## Files

- [model_container.py](model_container.md): Cross-validation model container
- [config.py](config.md): Cross-validation configuration and prediction modes
- [dataset.py](dataset.md): Cross-validation dataset handling
- [raw_model.py](raw_model.md): Cross-validation raw model container

## Key Classes

### Model Container

- [CvModelContainer](model_container.md#cvmodelcontainer): Main container for cross-validation models

### Configuration

- [CvLearner](config.md#cvlearner): Learner for cross-validation
- [OutOfFoldPredictor](config.md#outoffoldpredictor): Predictor for out-of-fold predictions
- [AverageEnsemblePredictor](config.md#averageensemblepredictor): Predictor for ensemble predictions

### Dataset

- [IndexedDataset](dataset.md#indexeddataset): Dataset with index information
- [IndexedDatasets](dataset.md#indexeddatasets): Collection of indexed datasets
- [SplittedDataset](dataset.md#splitteddataset): Dataset split into train, validation, and test sets
- [SplittedDatasets](dataset.md#splitteddatasets): Collection of split datasets

### Raw Model

- [RawModels](raw_model.md#rawmodels): Container for multiple raw models from cross-validation

## Usage Example

```python
from sklearn.model_selection import KFold
from factrainer.core import CvModelContainer
from factrainer.lightgbm import LgbDataset, LgbModelConfig, LgbTrainConfig

# Create dataset and configuration
dataset = LgbDataset(...)
config = LgbModelConfig.create(
    train_config=LgbTrainConfig(...),
)

# Set up cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)

# Get ensemble predictions for new data
new_dataset = LgbDataset(...)
y_pred_new = model.predict(new_dataset, n_jobs=4, mode=PredMode.AVG_ENSEMBLE)
