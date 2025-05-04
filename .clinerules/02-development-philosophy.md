# Development Philosophy

Our goal is to deliver value quickly to users while maintaining high software quality. We follow proven development methodologies (Agile and Extreme Programming) to iterate rapidly and improve continuously. Below are the core values, principles, and practices that Cline deems essential.

## Values

- **Rapid Delivery**: We prioritize providing useful features to users as quickly as possible.
- **Continuous Improvement**: We continually iterate and enhance features in line with the evolving machine learning ecosystem.

## Principles

- To realize the values above, we place the highest emphasis on software quality.
- Consequently, quick "workaround" fixes are forbidden. Always align with official documentation recommendations and ideal design practices. Even when a solution seems satisfactory for Cline, pause to reflect and find ways to improve.

## Practices

### Domain-Driven Design

- Express the domain and intent of the software clearly in code and tests. The code (including class and method names) and test cases should make it obvious what the software is trying to accomplish.
- Do **not** write inline comments in code. Code should be self-explanatory; relying on comments to clarify intent indicates the code isn't clear enough. Well-named classes, methods, and thorough tests should eliminate the need for most explanatory comments.

### SOLID Principles

- Always refer to the SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) when designing or refactoring. Aim to keep the system easy to extend and modify.
- Be careful not to let Factrainer’s design be guided by established ML OSS (e.g. scikit-learn). While they are valuable references for user API needs, their long history means they lack type safety and still use many imperative patterns.

### Test-Driven Development

- Develop using **Test-Driven Development (TDD)**: always begin by designing and writing tests before implementing the corresponding functionality.
- Follow an outside-in approach to testing: start with tests for the highest-level behavior (what the user wants to achieve), then write tests for lower-level components (functional tests, then unit tests) as needed. Write just enough code to make each failing test pass before moving on.
- Ensure tests initially fail for the expected reasons (red phase), then write implementation code to make them pass (green phase). Use refactoring (blue phase) to improve code guided by the tests.
- Don’t write tests first as a formality—let the tests drive your design decisions.
- Treat test code with the same level of care as production code. Strive for clear, maintainable tests that thoroughly cover the intended functionality.

### CI/CD

- Invest in automation for testing and deployment. A robust Continuous Integration/Continuous Deployment pipeline is crucial for reliability and developer productivity.
- Keep the feedback loop fast to enable short lead times and frequent releases:

  - Avoid fragile or flaky tests; tests should be reliable and deterministic.
  - Minimize the total test execution time; a faster test suite means quicker iteration and feedback.

### Trunk-Based Development

- We adopt trunk-based development to maximize development velocity. All changes are integrated frequently into the main branch.
- Trunk-based development requires solid CI/CD support and the ability to commit small, incremental changes often.
- In cases of large architectural changes or breaking feature development, a short-lived feature branch may be used, but the preference is to merge back to trunk quickly.
- _(Note: Trunk-based development works well given that currently there is a single primary developer. If the contributor base grows, we may re-evaluate this strategy and consider a branching model like GitHub Flow.)_
