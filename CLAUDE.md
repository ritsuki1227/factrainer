# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Install dependencies
```sh
# Install all packages with all extras and dev dependencies
uv sync --all-extras --all-groups

# Install specific framework support only
uv sync --extra lightgbm --all-groups
uv sync --extra sklearn --all-groups
```

### Run tests
```sh
# Run all tests
uv run pytest

# Run tests for specific package
uv run pytest tests/lightgbm
uv run pytest tests/sklearn

# Run specific test file
uv run pytest tests/lightgbm/unit/test_dataset.py

# Run tests with coverage
uv run pytest -v --cov --cov-report=xml
```

### Type checking and linting
```sh
# Run type checker
uv run mypy

# Run linter
uv run ruff check

# Check code formatting
uv run ruff format --check

# Fix formatting
uv run ruff format
```

## High-Level Architecture

Factrainer is a meta-framework for cross-validation across ML libraries, designed as a namespace package with the following structure:

### Core Components

1. **factrainer-base**: Foundation layer defining common interfaces and abstractions for plugin integration. Has no dependencies on other factrainer packages.

2. **factrainer-core**: Provides the main cross-validation logic and public API. Depends only on factrainer-base.

3. **Plugin packages** (factrainer-lightgbm, factrainer-sklearn, factrainer-xgboost, factrainer-catboost): Each plugin implements framework-specific functionality and depends only on factrainer-base. They inject their implementations into core's interfaces.

### Key Design Principles

- **Separation of Concerns**: Core CV logic is framework-agnostic; specific ML frameworks plug in as interchangeable components
- **Domain-Driven Design**: Code should be self-explanatory without comments
- **SOLID Principles**: Clean separation between what users want (CV) and how it's done (specific framework)
- **Test-Driven Development**: Write tests first, following outside-in approach (integration → functional → unit)

### Testing Structure

```
tests/
├── core/           # Unit tests for core logic
├── lightgbm/
│   ├── unit/       # Plugin-specific tests (no core dependency)
│   ├── functional/ # Plugin-core integration tests
│   └── integration/ # End-to-end user workflow tests
├── sklearn/        # Same structure as lightgbm
├── xgboost/        # Same structure as lightgbm
└── catboost/       # Same structure as lightgbm
```

### Development Workflow

1. Changes should be small and incremental (trunk-based development)
2. All code must pass type checking (mypy) and linting (ruff)
3. No inline comments - code should be self-documenting
4. Follow TDD: write failing test → implement → refactor
5. Definition of Done: passes all tests and static type checks

### Important Notes

- Python 3.12+ required (use new language features when appropriate)
- All configuration is in pyproject.toml files (root and per-package)
- Use `uv` for dependency management
- Optional dependencies keep installations lean - users install only needed plugins

## Git Command Policy

### Read-only Git commands (allowed)
- git status, git log, git diff, git show, git branch (list), git tag (list)
- git remote -v, git config --list, git describe, git rev-parse
- git ls-files, git blame

### State-changing Git commands (NOT allowed)
- git add, git commit, git push, git pull, git fetch, git merge, git rebase
- git checkout, git branch (create/delete), git tag (create/delete)
- git reset, git revert, git stash, git cherry-pick
- git config --global, git rm, git mv, git clean
- git remote add/remove, git submodule add/update
- Never modify repository state without explicit user request