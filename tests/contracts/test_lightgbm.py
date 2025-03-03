from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import lightgbm as lgb
import pytest


def test_train_signature() -> None:
    expected = Signature(
        [
            Parameter(
                "params", Parameter.POSITIONAL_OR_KEYWORD, annotation=Dict[str, Any]
            ),
            Parameter(
                "train_set", Parameter.POSITIONAL_OR_KEYWORD, annotation=lgb.Dataset
            ),
            Parameter(
                "num_boost_round",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
                default=100,
            ),
            Parameter(
                "valid_sets",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[List[lgb.Dataset]],
                default=None,
            ),
            Parameter(
                "valid_names",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[List[str]],
                default=None,
            ),
            Parameter(
                "feval",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[
                    Union[
                        lgb.engine._LGBM_CustomMetricFunction,
                        List[lgb.engine._LGBM_CustomMetricFunction],
                    ]
                ],
                default=None,
            ),
            Parameter(
                "init_model",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[Union[str, Path, lgb.Booster]],
                default=None,
            ),
            Parameter(
                "keep_training_booster",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter(
                "callbacks",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[List[Callable]],  # type: ignore
                default=None,
            ),
        ],
        return_annotation=lgb.Booster,
    )
    actual = signature(lgb.train)

    assert actual == expected


@pytest.mark.skip
def test_lgbm_train_data_type() -> None:
    # actual = _LGBM_TrainDataType.__args__
    raise NotImplementedError
