# What is Factrainer?

Factrainer is an open-source machine learning library in Python that serves as a meta-framework for cross-validation across various ML frameworks.

- **Unified Cross-Validation API**: It provides a unified API for performing cross-validation (CV) using frameworks like scikit-learn, LightGBM, XGBoost, and CatBoost. Through Factrainer's API, you can run CV with any of these frameworks, making it a higher-level abstraction over them.
- **Meta-Framework Capabilities**: Factrainer's design allows it to manage CV across different libraries seamlessly. Users invoke Factrainer for CV, and under the hood it interfaces with the chosen ML framework.
- **Enhanced CV Functionality**: It offers features that are not well-supported by existing ML libraries, such as:
  - Keeping trained model objects from each CV fold.
  - Parallel execution of CV folds.
  - Simplified generation of out-of-fold (OOF) predictions via a clean API.
- _(For more details, refer to the project’s root `README.md`.)_

## Intended Users

Factrainer is intended for:

- Industry data scientists and machine learning engineers.
- Academic researchers and anyone who needs a consistent cross-validation workflow across multiple ML libraries.
