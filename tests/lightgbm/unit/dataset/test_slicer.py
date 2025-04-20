from typing import Any
from unittest.mock import MagicMock, patch

import lightgbm as lgb
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


class TestDatasetSlicer:
    @patch(
        "factrainer.lightgbm.dataset.slicer.LgbPositionSlicer", spec=LgbPositionSlicer
    )
    @patch(
        "factrainer.lightgbm.dataset.slicer.LgbInitScoreSlicer", spec=LgbInitScoreSlicer
    )
    @patch("factrainer.lightgbm.dataset.slicer.LgbGroupSlicer", spec=LgbGroupSlicer)
    @patch("factrainer.lightgbm.dataset.slicer.LgbWeightSlicer", spec=LgbWeightSlicer)
    @patch("factrainer.lightgbm.dataset.slicer.LgbLabelSlicer", spec=LgbLabelSlicer)
    @patch("factrainer.lightgbm.dataset.slicer.LgbDataSlicer", spec=LgbDataSlicer)
    def test_dataset_slicer(
        self,
        data_slicer: MagicMock,
        label_slicer: MagicMock,
        weight_slicer: MagicMock,
        group_slicer: MagicMock,
        init_score_slicer: MagicMock,
        position_slicer: MagicMock,
    ) -> None:
        dataset = MagicMock(spec=lgb.Dataset)
        dataset.data = MagicMock(spec=_LGBM_TrainDataType)
        dataset.label = MagicMock(spec=_LGBM_LabelType)
        dataset.weight = MagicMock(spec=_LGBM_WeightType)
        dataset.group = MagicMock(spec=_LGBM_GroupType)
        dataset.init_score = MagicMock(spec=_LGBM_InitScoreType)
        dataset.feature_name = MagicMock(spec=_LGBM_FeatureNameConfiguration)
        dataset.categorical_feature = MagicMock(
            spec=_LGBM_CategoricalFeatureConfiguration
        )
        dataset.params = MagicMock(spec=dict[str, Any])
        dataset.free_raw_data = MagicMock(spec=bool)
        dataset.raw_data = MagicMock(spec=str)
        dataset.position = MagicMock(spec=_LGBM_PositionType)
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
        assert actual.weight == expected.weight
        assert actual.group == expected.group
        assert actual.init_score == expected.init_score
        assert actual.feature_name == expected.feature_name
        assert actual.categorical_feature == expected.categorical_feature
        assert actual.free_raw_data == expected.free_raw_data
        assert actual.position == expected.position
        assert expected.reference is None
        data_slicer.return_value.slice.assert_called_once_with(dataset.data, [2, 0])
        label_slicer.return_value.slice.assert_called_once_with(dataset.label, [2, 0])
        weight_slicer.return_value.slice.assert_called_once_with(dataset.weight, [2, 0])
        group_slicer.return_value.slice.assert_called_once_with(dataset.group, [2, 0])
        init_score_slicer.return_value.slice.assert_called_once_with(
            dataset.init_score, [2, 0]
        )
        position_slicer.return_value.slice.assert_called_once_with(
            dataset.position, [2, 0]
        )
