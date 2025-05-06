# factrainer.core

The `factrainer.core` module provides the main functionality for cross-validation in Factrainer. It contains the model containers and cross-validation utilities that form the public API of the library.

## Public API

The following classes and functions are part of the public API of the `factrainer.core` module:

- [CvModelContainer](cvmodelcontainer.md): Container for cross-validation models
- [SingleModelContainer](singlemodelcontainer.md): Container for single models
- [PredMode](predmode.md): Enumeration of prediction modes
- [SplittedDatasetsIndices](splitteddatasetsindices.md): Cross-validation dataset indices

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
        params={"objective": "regression"},
        callbacks=[lgb.early_stopping(100, verbose=False)],
    ),
)

# Set up cross-validation
k_fold = KFold(n_splits=4, shuffle=True, random_state=1)

# Create and train model
model = CvModelContainer(config, k_fold)
model.train(dataset, n_jobs=4)

# Get OOF predictions
y_pred = model.predict(dataset, n_jobs=4)
