import numpy as np
import pandas as pd
import pytest
from lightgbm.basic import (
    _LGBM_FeatureNameConfiguration,
    _LGBM_GroupType,
    _LGBM_LabelType,
    _LGBM_TrainDataType,
    _LGBM_WeightType,
)

from factrainer.lightgbm.dataset.equality_checker import (
    LgbDataEqualityChecker,
    LgbFeatureNameEqualityChecker,
    LgbGroupEqualityChecker,
    LgbLabelEqualityChecker,
    LgbWeightEqualityChecker,
)


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            np.array([[100, 10], [200, 20]]),
            pd.DataFrame(np.array([[100, 10], [200, 20]])),
            False,
            id="negative/different-type",
        ),
        pytest.param(
            np.array([[100, 10], [200, 20]]),
            np.array([[100, 10], [200, 20]]),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            np.array([[100, 10], [200, 20]]),
            np.array([[100, 10], [200, 99]]),
            False,
            id="negative/numpy",
        ),
        pytest.param(
            pd.DataFrame(np.array([[100, 10], [200, 20]])),
            pd.DataFrame(np.array([[100, 10], [200, 20]])),
            True,
            id="normal/pandas",
        ),
        pytest.param(
            pd.DataFrame(np.array([[100, 10], [200, 20]])),
            pd.DataFrame(np.array([[100, 10], [200, 99]])),
            False,
            id="negative/pandas",
        ),
    ],
)
def test_lgb_data_equality_checker(
    left: _LGBM_TrainDataType, right: _LGBM_TrainDataType, expected: bool
) -> None:
    sut = LgbDataEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            None,
            None,
            True,
            id="normal/none",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series([1, 2, 3]),
            False,
            id="negative/different-type",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 999]),
            False,
            id="negative/numpy",
        ),
        pytest.param(
            pd.DataFrame(np.array([[1], [2], [3]])),
            pd.DataFrame(np.array([[1], [2], [3]])),
            True,
            id="normal/pandas/df",
        ),
        pytest.param(
            pd.DataFrame(np.array([[1], [2], [3]])),
            pd.DataFrame(np.array([[1], [2], [999]])),
            False,
            id="negative/pandas/df",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            True,
            id="normal/pandas/series",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 999]),
            False,
            id="negative/pandas/series",
        ),
    ],
)
def test_lgb_label_equality_checker(
    left: _LGBM_LabelType | None, right: _LGBM_LabelType | None, expected: bool
) -> None:
    sut = LgbLabelEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            None,
            None,
            True,
            id="normal/none",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series([1, 2, 3]),
            False,
            id="negative/different-type",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 999]),
            False,
            id="negative/numpy",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            True,
            id="normal/pandas/series",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 999]),
            False,
            id="negative/pandas/series",
        ),
    ],
)
def test_lgb_weight_equality_checker(
    left: _LGBM_WeightType | None, right: _LGBM_WeightType | None, expected: bool
) -> None:
    sut = LgbWeightEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            None,
            None,
            True,
            id="normal/none",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series([1, 2, 3]),
            False,
            id="negative/different-type",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 999]),
            False,
            id="negative/numpy",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            True,
            id="normal/pandas/series",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 999]),
            False,
            id="negative/pandas/series",
        ),
    ],
)
def test_lgb_init_score_equality_checker(
    left: _LGBM_GroupType | None, right: _LGBM_GroupType | None, expected: bool
) -> None:
    sut = LgbGroupEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            None,
            None,
            True,
            id="normal/none",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series([1, 2, 3]),
            False,
            id="negative/different-type",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            True,
            id="normal/numpy",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.array([1, 2, 999]),
            False,
            id="negative/numpy",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            True,
            id="normal/pandas/series",
        ),
        pytest.param(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 999]),
            False,
            id="negative/pandas/series",
        ),
    ],
)
def test_lgb_group_equality_checker(
    left: _LGBM_GroupType | None, right: _LGBM_GroupType | None, expected: bool
) -> None:
    sut = LgbGroupEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected


@pytest.mark.parametrize(
    ["left", "right", "expected"],
    [
        pytest.param(
            "auto",
            "auto",
            True,
            id="normal/auto",
        ),
        pytest.param(
            "auto",
            ["test"],
            False,
            id="negative/different-type",
        ),
        pytest.param(
            ["feature_0", "feature_1"],
            ["feature_0", "feature_1"],
            True,
            id="normal/features",
        ),
    ],
)
def test_lgb_feature_name_equality_checker(
    left: _LGBM_FeatureNameConfiguration,
    right: _LGBM_FeatureNameConfiguration,
    expected: bool,
) -> None:
    sut = LgbFeatureNameEqualityChecker()

    actual = sut.check(left, right)

    assert actual is expected
