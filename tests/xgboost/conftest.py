from collections.abc import Sequence
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    for item in items:
        if THIS_DIR in Path(item.fspath).parents:
            item.add_marker(pytest.mark.xgboost)
