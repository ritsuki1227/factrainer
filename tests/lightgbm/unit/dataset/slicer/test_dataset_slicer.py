from typing import Any
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
from factrainer.lightgbm.dataset.slicer import (
    LgbDatasetSlicer,
    LgbDataSlicer,
    LgbGroupSlicer,
    LgbInitScoreSlicer,
    LgbLabelSlicer,
    LgbPositionSlicer,
    LgbWeightSlicer,
)
from lightgbm.basic import (
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
    _LGBM_GroupType,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_PositionType,
    _LGBM_TrainDataType,
    _LGBM_WeightType,
)
from numpy.testing import assert_array_equal


@patch("factrainer.lightgbm.dataset.slicer.LgbDataSlicer", spec=LgbDataSlicer)
def test_dataset_slicer(data_slicer: MagicMock) -> None:
    data = MagicMock(spec=_LGBM_TrainDataType)
    dataset = lgb.Dataset(data)
    expected = lgb.Dataset(
        data=data_slicer.return_value.slice.return_value,
    )
    sut = LgbDatasetSlicer()

    actual = sut.slice(dataset, [2, 0])

    assert actual.data == expected.data
    assert actual.label is None
    assert expected.reference is None
    assert actual.weight is None
    assert actual.group is None
    assert actual.init_score is None
    assert actual.feature_name == "auto"
    assert actual.categorical_feature == "auto"
    assert actual.free_raw_data is True
    data_slicer.return_value.slice.assert_called_once_with(data, [2, 0])


@patch("factrainer.lightgbm.dataset.slicer.LgbPositionSlicer", spec=LgbPositionSlicer)
@patch("factrainer.lightgbm.dataset.slicer.LgbInitScoreSlicer", spec=LgbInitScoreSlicer)
@patch("factrainer.lightgbm.dataset.slicer.LgbGroupSlicer", spec=LgbGroupSlicer)
@patch("factrainer.lightgbm.dataset.slicer.LgbWeightSlicer", spec=LgbWeightSlicer)
@patch("factrainer.lightgbm.dataset.slicer.LgbLabelSlicer", spec=LgbLabelSlicer)
@patch("factrainer.lightgbm.dataset.slicer.LgbDataSlicer", spec=LgbDataSlicer)
def test_dataset_slicer_with_params(
    data_slicer: MagicMock,
    label_slicer: MagicMock,
    weight_slicer: MagicMock,
    group_slicer: MagicMock,
    init_score_slicer: MagicMock,
    position_slicer: MagicMock,
) -> None:
    dataset = lgb.Dataset(
        data=MagicMock(spec=_LGBM_TrainDataType),
        label=MagicMock(spec=_LGBM_LabelType),
        weight=MagicMock(spec=_LGBM_WeightType),
        group=MagicMock(spec=_LGBM_GroupType),
        init_score=MagicMock(spec=_LGBM_InitScoreType),
        feature_name=MagicMock(spec=_LGBM_FeatureNameConfiguration),
        categorical_feature=MagicMock(spec=_LGBM_CategoricalFeatureConfiguration),
        params=MagicMock(spec=dict[str, Any]),
        free_raw_data=MagicMock(spec=bool),
        position=MagicMock(spec=_LGBM_PositionType),
    )
    expected = lgb.Dataset(
        data=data_slicer.return_value.slice.return_value,
        label=label_slicer.return_value.slice.return_value,
        reference=None,
        weight=weight_slicer.return_value.slice.return_value,
        group=group_slicer.return_value.slice.return_value,
        init_score=init_score_slicer.return_value.slice.return_value,
        feature_name=dataset.feature_name,
        categorical_feature=dataset.categorical_feature,
        params=dataset.params,
        free_raw_data=dataset.free_raw_data,
        position=position_slicer.return_value.slice.return_value,
    )
    sut = LgbDatasetSlicer()

    actual = sut.slice(dataset, [2, 0])

    assert actual.data == expected.data
    assert actual.label == expected.label
    assert expected.reference is None
    assert actual.weight == expected.weight
    assert actual.group == expected.group
    assert actual.init_score == expected.init_score
    assert actual.feature_name == expected.feature_name
    assert actual.categorical_feature == expected.categorical_feature
    assert actual.free_raw_data == expected.free_raw_data
    assert actual.position == expected.position
    data_slicer.return_value.slice.assert_called_once_with(dataset.data, [2, 0])
    label_slicer.return_value.slice.assert_called_once_with(dataset.label, [2, 0])
    weight_slicer.return_value.slice.assert_called_once_with(dataset.weight, [2, 0])
    group_slicer.return_value.slice.assert_called_once_with(dataset.group, [2, 0])
    init_score_slicer.return_value.slice.assert_called_once_with(
        dataset.init_score, [2, 0]
    )
    position_slicer.return_value.slice.assert_called_once_with(dataset.position, [2, 0])


def test_dataset_slicer_with_reference() -> None:
    dataset = lgb.Dataset(np.array([[100, 200], [300, 400]]))
    reference = lgb.Dataset(np.array([[1, 2], [3, 4]]))
    expected = lgb.Dataset(np.array([[1, 2], [3, 4]]))
    sut = LgbDatasetSlicer(reference=reference)

    actual = sut.slice(dataset, [1])

    if actual.reference is None:
        raise TypeError
    if not isinstance(actual.reference.data, np.ndarray):
        raise TypeError
    if not isinstance(expected.data, np.ndarray):
        raise TypeError
    assert_array_equal(actual.reference.data, expected.data)
