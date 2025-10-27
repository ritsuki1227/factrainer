#!/bin/bash
set -e

uv run mypy
uv run pytest -v --cov --cov-report=xml
uv run ruff format --check
uv run ruff check