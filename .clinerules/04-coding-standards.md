# Coding Standards

- **Python Version**: Use Python 3.12 or above. Take advantage of new language features introduced in Python 3.12 (for example, PEP 695’s syntax for type parameters in generics) when appropriate.
- **Style Guidelines**: Adhere to standard Python style conventions (PEP 8). Code formatting and linting are automatically enforced via `ruff`.
- **Testing Practices**: Write tests (using `pytest`) to verify that code behaves as expected and that expected errors are raised when appropriate.
- **"Definition of Done"**: A code change is only considered complete when it passes all tests and static type checks (`mypy`). Never consider an implementation finished without running and passing the full test suite and type checker.
- **No Comments**: - Do not write comments in production or test files. Code and tests must speak for themselves via descriptive names and well-structured logic.
- **Git Usage**: Cline is allow to run read-only Git commands (e.g. git status, git diff) only. Do not execute commands that modify repository state (such as git add or git commit).
