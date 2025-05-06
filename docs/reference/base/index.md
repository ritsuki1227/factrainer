# factrainer.base

The `factrainer.base` module provides the core interfaces and abstractions for Factrainer. It defines the plugin architecture that allows different ML frameworks to be integrated into Factrainer.

## Module Purpose

This module contains the base interfaces that are implemented by the framework-specific plugins:

- Base dataset interfaces
- Base configuration interfaces
- Base model interfaces

These interfaces define the contract that must be followed by all framework implementations, enabling Factrainer to work with different ML frameworks in a consistent way.

## Plugin Architecture

Factrainer uses a plugin architecture to support different ML frameworks. Each framework implements the interfaces defined in `factrainer.base` to provide framework-specific functionality.

The plugin architecture allows Factrainer to:

1. Provide a unified API for cross-validation across different frameworks
2. Support framework-specific features and optimizations
3. Allow users to choose which frameworks to install and use

## Implementation Example

To implement a new plugin for a framework, you need to:

1. Create classes that implement the base interfaces
2. Register the plugin with Factrainer

```python
from factrainer.base.config import BaseLearner, BasePredictor, MlModelConfig
from factrainer.base.dataset import IndexableDataset
from factrainer.base.raw_model import RawModel

# 1. Define the raw model wrapper
class MyFrameworkModel(RawModel):
    model: Any  # The actual model from the framework

# 2. Define the dataset wrapper
class MyFrameworkDataset(IndexableDataset):
    dataset: Any  # The actual dataset from the framework
    
    def __getitem__(self, index):
        # Implement indexing
        ...
    
    def k_fold_split(self, k_fold):
        # Implement k-fold splitting
        ...

# 3. Define the configuration classes
class MyFrameworkTrainConfig(BaseTrainConfig):
    # Training parameters
    ...

class MyFrameworkPredictConfig(BasePredictConfig):
    # Prediction parameters
    ...

# 4. Define the learner and predictor
class MyFrameworkLearner(BaseLearner[MyFrameworkDataset, MyFrameworkModel, MyFrameworkTrainConfig]):
    def train(self, train_dataset, val_dataset, config):
        # Implement training
        ...

class MyFrameworkPredictor(BasePredictor[MyFrameworkDataset, MyFrameworkModel, MyFrameworkPredictConfig]):
    def predict(self, dataset, raw_model, config):
        # Implement prediction
        ...

# 5. Define the model configuration
class MyFrameworkModelConfig(MlModelConfig[MyFrameworkDataset, MyFrameworkModel, MyFrameworkTrainConfig, MyFrameworkPredictConfig]):
    @classmethod
    def create(cls, train_config, pred_config=None):
        return cls(
            learner=MyFrameworkLearner(),
            predictor=MyFrameworkPredictor(),
            train_config=train_config,
            pred_config=pred_config or MyFrameworkPredictConfig(),
        )
