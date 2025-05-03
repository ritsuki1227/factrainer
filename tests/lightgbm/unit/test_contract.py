from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import scipy
from lightgbm import Dataset
from lightgbm.basic import (
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
    _LGBM_GroupType,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_PositionType,
    _LGBM_PredictDataType,
    _LGBM_TrainDataType,
    _LGBM_WeightType,
)


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


def test_predict_signature() -> None:
    expected = Signature(
        [
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter(
                "data",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=_LGBM_PredictDataType,
            ),
            Parameter(
                "start_iteration",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
                default=0,
            ),
            Parameter(
                "num_iteration",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[int],
                default=None,
            ),
            Parameter(
                "raw_score",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter(
                "pred_leaf",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter(
                "pred_contrib",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter(
                "data_has_header",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter(
                "validate_features",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=False,
            ),
            Parameter("kwargs", Parameter.VAR_KEYWORD, annotation=Any),
        ],
        return_annotation=Union[
            np.ndarray, scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]
        ],
    )

    actual = signature(lgb.Booster.predict)

    assert actual == expected


def test_dataset_signature() -> None:
    expected = Signature(
        [
            Parameter(
                "data",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=_LGBM_TrainDataType,
            ),
            Parameter(
                "label",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[_LGBM_LabelType],
                default=None,
            ),
            Parameter(
                "reference",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional["Dataset"],
                default=None,
            ),
            Parameter(
                "weight",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[_LGBM_WeightType],
                default=None,
            ),
            Parameter(
                "group",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[_LGBM_GroupType],
                default=None,
            ),
            Parameter(
                "init_score",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[_LGBM_InitScoreType],
                default=None,
            ),
            Parameter(
                "feature_name",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=_LGBM_FeatureNameConfiguration,
                default="auto",
            ),
            Parameter(
                "categorical_feature",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=_LGBM_CategoricalFeatureConfiguration,
                default="auto",
            ),
            Parameter(
                "params",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[Dict[str, Any]],
                default=None,
            ),
            Parameter(
                "free_raw_data",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=bool,
                default=True,
            ),
            Parameter(
                "position",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[_LGBM_PositionType],
                default=None,
            ),
        ],
    )
    actual = signature(lgb.Dataset)
    assert actual == expected
