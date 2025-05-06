# API Reference

Welcome to the Factrainer API Reference. This section provides detailed documentation for all public modules, classes, and functions in the Factrainer library.

## Package Structure

Factrainer is organized as a namespace package with several sub-packages:

- **[factrainer.base](base/index.md)**: Core interfaces and abstractions that define the plugin architecture
- **[factrainer.core](core/index.md)**: Main cross-validation functionality and public API
- **[factrainer.lightgbm](lightgbm/index.md)**: LightGBM integration
- **[factrainer.sklearn](sklearn/index.md)**: scikit-learn integration
- **[factrainer.xgboost](xgboost/index.md)**: XGBoost integration (coming soon)
- **[factrainer.catboost](catboost/index.md)**: CatBoost integration (coming soon)

## Public API

This reference documents only the public API of Factrainer, which consists of the objects explicitly exported in each sub-package's `__init__.py` file. These are the objects that are intended to be used by users of the library.

### Core API

- [CvModelContainer](core/cvmodelcontainer.md): Container for cross-validation models
- [SingleModelContainer](core/singlemodelcontainer.md): Container for single models
- [PredMode](core/predmode.md): Enumeration of prediction modes
- [SplittedDatasetsIndices](core/splitteddatasetsindices.md): Cross-validation dataset indices

### LightGBM API

- [LgbDataset](lightgbm/lgbdataset.md): LightGBM dataset wrapper
- [LgbModelConfig](lightgbm/lgbmodelconfig.md): Configuration for LightGBM models
- [LgbTrainConfig](lightgbm/lgbtrainconfig.md): Configuration for LightGBM training
- [LgbPredictConfig](lightgbm/lgbpredictconfig.md): Configuration for LightGBM prediction
- [LgbModel](lightgbm/lgbmodel.md): LightGBM model wrapper

### scikit-learn API

- [SklearnDataset](sklearn/sklearndataset.md): scikit-learn dataset wrapper
- [SklearnModelConfig](sklearn/sklearnmodelconfig.md): Configuration for scikit-learn models
- [SklearnTrainConfig](sklearn/sklearntrainconfig.md): Configuration for scikit-learn training
- [SklearnPredictConfig](sklearn/sklearnpredictconfig.md): Configuration for scikit-learn prediction
- [SklearnPredictMethod](sklearn/sklearnpredictmethod.md): Enumeration of scikit-learn prediction methods
- [SklearnModel](sklearn/sklearnmodel.md): scikit-learn model wrapper
