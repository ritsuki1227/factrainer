# Software Architecture

## Overview

- **Meta-Framework Design**: Factrainer serves as a meta-framework that allows users to perform cross-validation (CV) using different ML libraries (scikit-learn, LightGBM, XGBoost, CatBoost, etc) through a unified interface. Users call Factrainer’s API to run CV, and Factrainer handles interfacing with the underlying framework.
- **Abstraction and Implementation**: For instance, CV is treated as the high-level abstraction (the feature the user wants), and the specific ML frameworks are the implementations. In line with SOLID principles, Factrainer cleanly separates what the user wants to do (perform CV) from how it’s done (via a specific library).
- **Layered Architecture**: The system is structured into an abstract core layer and multiple plug-in layers for each supported framework. This separation ensures that core logic (e.g. CV) is framework-agnostic, and frameworks can plug into the system as interchangeable components.

## Core Architecture

Factrainer is organized as a **namespace package** (per PEP 420) composed of several sub-packages:

- **factrainer-base**: Defines common interfaces and abstractions (especially for plugin integration). This base layer has no dependencies on any other factrainer sub-package (core or plugins).
- **factrainer-core**: Provides the core cross-validation logic and public API. Depends on factrainer-base but not on any other factrainer plugin sub-package.
- **factrainer-lightgbm**: Plugin layer for LightGBM. Depends on factrainer-base (but not on core). By injecting LightGBM-specific implementations into factrainer-core’s interfaces, it enables using LightGBM for CV.
- **factrainer-sklearn**: Plugin layer for scikit-learn. Depends on factrainer-base but not on core. Its implementation is injected into the core API to enable CV with scikit-learn.
- **factrainer-xgboost**: Plugin layer for XGBoost. Depends on factrainer-base but not on core. Injects XGBoost functionality into the core API to enable CV with XGBoost.
- **factrainer-catboost**: Plugin layer for CatBoost. Depends on factrainer-base but not on core. Injects CatBoost functionality into the core API to enable CV with CatBoost.
- _Optional Dependencies_: Per the project’s `pyproject.toml`, these plugin packages are **optional**. The factrainer-core package is always installed by default, and users can install only the specific plugin(s) they need. This keeps the installed dependencies lean.
- _Clean Architecture Note_: While the design follows SOLID principles, it does not strictly adhere to a pure Clean Architecture. We allow certain domain objects to wrap framework-specific data structures (e.g., NumPy arrays, pandas DataFrames, LightGBM Datasets). In the ML context, some departures from Clean Architecture are acceptable to meet performance and usability requirements.

## Dependency Management

- We use **uv** to manage Python environments and dependencies.
- All project configuration (dependencies, tool settings for mypy/pytest/ruff, etc.) is declared in `pyproject.toml`. Because Factrainer is a namespace package with multiple sub-packages, there is a `pyproject.toml` at the project root and one in each sub-package directory.
- To install the entire project with all optional components and development tools, use:

```sh
uv sync --all-extras --all-groups
```

This installs all plugin extras and dev dependencies in one go.

## Testing

### Overview

- `uv run pytest` – run all tests.
- `uv run mypy` – run the static type checker.
- `uv run ruff check` – run lint checks.
- `uv run ruff format` --check – verify code formatting.

Configuration for `pytest`, `mypy`, and `ruff` is specified in the root `pyproject.toml`.

### Structure

- The tests/ directory is organized by sub-package:
  - `tests/core/` – contains unit tests for the factrainer-core module (core logic).
  - `tests/lightgbm/`, `tests/sklearn/`, `tests/xgboost/`, `tests/catboost/` – contain tests for each plugin package.
    - Within each plugin’s test folder:
      - `unit/` – unit tests for that plugin’s internal logic (these tests **should not** depend on the core package).
      - `functional/` – tests for the plugin’s integration with the core API (these tests **do** depend on core; they verify that injecting the plugin into the core works as expected).
      - `integration/` – end-to-end tests simulating how a user uses Factrainer with that plugin (these also depend on core, testing the full workflow).
    - This testing hierarchy ensures separation of concerns:
      - Unit tests validate plugin-specific behavior in isolation.
      - Functional tests validate the interaction between each plugin and the core logic.
      - Integration tests validate that a user can use Factrainer (core + plugin) to perform cross-validation as intended.

## CI/CD Pipeline

- We use **GitHub Actions** for Continuous Integration and Continuous Deployment.
- Each commit to the repository triggers a CI workflow. (Our workflow files are structured so that one workflow runs per commit, encompassing all checks.)
- **Release Automation**: Two workflow files manage the release process:
  - `cd-trigger.yaml`: Manually triggered and it checks out the release branch, increments the version number in `pyproject.toml`, and commits the change.
  - `cd-release.yaml`: Runs after CI passes on the release branch. This workflow tags the commit and publishes the package to GitHub Releases and PyPI.

This setup ensures every change is validated by tests, and releases are performed in a controlled, automated fashion.

## Documentation

- User-facing documentation is planned to be written using MkDocs. _(As of now, the documentation site is not yet implemented, but it’s on the roadmap.)_
